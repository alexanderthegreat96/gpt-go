package gpt

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// synthetic gpt-2 dimensions. tiny on purpose -> we are testing the wiring,
// not the maths, and a 4 wide model exercises every code path a 768 wide one
// would while staying readable in a failure message
const (
	testVocabSize = 10
	testDModel    = 4
	testNumHeads  = 2
	testNumLayers = 2
	testMaxSeqLen = 8
	testDInner    = 4 * testDModel
)

// buildSyntheticGPT2 writes a checkpoint shaped exactly like huggingface gpt-2,
// with the given tensor name prefix. values are deterministic but arbitrary -> we
// only care that each one lands in the right place
func buildSyntheticGPT2(t *testing.T, prefix string) string {
	t.Helper()

	tensors := map[string][]float32{}
	shapes := map[string][]int{}

	add := func(name string, shape ...int) {
		count := 1
		for _, d := range shape {
			count *= d
		}
		values := make([]float32, count)
		for i := range values {
			// something non-zero and non-uniform, so a misplaced tensor shows
			// up as a wrong number rather than coincidentally matching
			values[i] = float32(i%7)*0.1 - 0.3
		}
		tensors[prefix+name] = values
		shapes[prefix+name] = shape
	}

	add("wte.weight", testVocabSize, testDModel)
	add("wpe.weight", testMaxSeqLen, testDModel)

	for i := 0; i < testNumLayers; i++ {
		base := fmt.Sprintf("h.%d.", i)
		add(base+"ln_1.weight", testDModel)
		add(base+"ln_1.bias", testDModel)
		// the fused qkv, three times the hidden dim wide
		add(base+"attn.c_attn.weight", testDModel, 3*testDModel)
		add(base+"attn.c_attn.bias", 3*testDModel)
		add(base+"attn.c_proj.weight", testDModel, testDModel)
		add(base+"attn.c_proj.bias", testDModel)
		add(base+"ln_2.weight", testDModel)
		add(base+"ln_2.bias", testDModel)
		add(base+"mlp.c_fc.weight", testDModel, testDInner)
		add(base+"mlp.c_fc.bias", testDInner)
		add(base+"mlp.c_proj.weight", testDInner, testDModel)
		add(base+"mlp.c_proj.bias", testDModel)
	}

	add("ln_f.weight", testDModel)
	add("ln_f.bias", testDModel)

	return writeTestSafeTensors(t, tensors, shapes, nil)
}

func TestLoadGPT2Shape(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	model, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	vocabSize, dModel, numLayers, numHeads, maxSeqLen := model.Describe()

	if vocabSize != testVocabSize {
		t.Errorf("expected vocab size %d, got %d", testVocabSize, vocabSize)
	}
	if dModel != testDModel {
		t.Errorf("expected dModel %d, got %d", testDModel, dModel)
	}
	if numLayers != testNumLayers {
		t.Errorf("expected %d layers, got %d", testNumLayers, numLayers)
	}
	if numHeads != testNumHeads {
		t.Errorf("expected %d heads, got %d", testNumHeads, numHeads)
	}
	if maxSeqLen != testMaxSeqLen {
		t.Errorf("expected context length %d, got %d", testMaxSeqLen, maxSeqLen)
	}

	// no lm_head.weight in the file, so it has to come from the embeddings
	if !model.IsLMHeadTied() {
		t.Errorf("expected the lm head to be tied to the token embeddings")
	}
	if model.lmHead.rows != testDModel || model.lmHead.cols != testVocabSize {
		t.Errorf("expected the lm head to be %dx%d, got %dx%d", testDModel, testVocabSize, model.lmHead.rows, model.lmHead.cols)
	}

	// every block should have its biases filled in and the tanh GELU selected
	for i, block := range model.blocks {
		if block.attention.bq == nil || block.attention.bk == nil || block.attention.bv == nil || block.attention.bo == nil {
			t.Errorf("block %d is missing attention biases", i)
		}
		if block.ffn.b1 == nil || block.ffn.b2 == nil {
			t.Errorf("block %d is missing mlp biases", i)
		}
		if !block.ffn.useTanhGELU {
			t.Errorf("block %d is not using the tanh GELU, pretrained weights expect it", i)
		}
		if block.ffn.dInner != testDInner {
			t.Errorf("block %d has inner dim %d, expected %d", i, block.ffn.dInner, testDInner)
		}
	}
}

func TestLoadGPT2DetectsTransformerPrefix(t *testing.T) {
	// a checkpoint saved from GPT2LMHeadModel prefixes everything, one saved
	// from GPT2Model does not. we should handle either without being told
	path := buildSyntheticGPT2(t, "transformer.")

	model, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error with a prefixed checkpoint: %v", err)
	}

	if _, _, numLayers, _, _ := model.Describe(); numLayers != testNumLayers {
		t.Errorf("expected %d layers, got %d", testNumLayers, numLayers)
	}
}

func TestLoadGPT2RejectsBadPrefix(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	// a typo in an explicit prefix should fail right away rather than twenty
	// tensors later with a confusing message
	if _, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads, Prefix: "wrong."}); err == nil {
		t.Errorf("expected an error for a prefix that is not in the file")
	}
}

func TestLoadGPT2RequiresHeadsForUnknownSize(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	// dModel of 4 is not a published gpt-2 size, so inference should refuse
	// rather than guess. for a real checkpoint (768, 1024, 1280, 1600) it works
	if _, err := LoadGPT2(path, GPT2Config{}); err == nil {
		t.Errorf("expected an error when the head count cannot be inferred")
	}
}

func TestLoadedGPT2RunsAForwardPass(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	model, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3}

	logits, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	if logits.rows != len(tokenIDs) || logits.cols != testVocabSize {
		t.Errorf("expected logits of %dx%d, got %dx%d", len(tokenIDs), testVocabSize, logits.rows, logits.cols)
	}

	// and nothing should have come out as NaN -> that usually means a
	// layernorm divided by zero or a bias landed somewhere it does not fit
	for i, v := range logits.data {
		if v != v {
			t.Fatalf("logit at index %d is NaN", i)
		}
	}
}

func TestCollectActivationsCoversEveryPoint(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	model, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	order, points, err := CollectActivations(model, []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// embed + ln_f + logits, plus six points per block
	expectedCount := 3 + 6*testNumLayers
	if len(order) != expectedCount {
		t.Errorf("expected %d activation points, got %d", expectedCount, len(order))
	}

	// the names have to line up with what the python dumper writes, so spot
	// check the ones that are easiest to get wrong
	required := []string{"embed", "h.0.ln_1", "h.0.attn", "h.0.res1", "h.0.ln_2", "h.0.mlp", "h.0.out", "ln_f", "logits"}
	for _, name := range required {
		m, ok := points[name]
		if !ok {
			t.Errorf("activation point %q is missing", name)
			continue
		}
		if m == nil {
			t.Errorf("activation point %q is nil", name)
		}
	}

	// the order matters -> it is what lets the cli say "this is the FIRST
	// divergence" instead of dumping an unordered pile of diffs
	if order[0] != "embed" {
		t.Errorf("expected the first point to be embed, got %q", order[0])
	}
	if order[len(order)-1] != "logits" {
		t.Errorf("expected the last point to be logits, got %q", order[len(order)-1])
	}

	// res1 really should be the block input plus the attention output, which
	// is how the python side reconstructs it
	embed := points["embed"]
	attn := points["h.0.attn"]
	res1 := points["h.0.res1"]
	for i := range res1.data {
		expected := embed.data[i] + attn.data[i]
		diff := expected - res1.data[i]
		if diff < -1e-5 || diff > 1e-5 {
			t.Fatalf("res1 at index %d: expected %f, got %f", i, expected, res1.data[i])
		}
	}
}

func TestLoadedGPT2SurvivesTheJSONRoundTrip(t *testing.T) {
	// saving a pretrained model to our json format and reading it back has to
	// preserve the biases and the activation choice. if it does not, the
	// reloaded model runs fine and produces different output, which is the
	// exact failure mode this whole chunk of work is built to prevent
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3}
	before, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	path := filepath.Join(t.TempDir(), "roundtrip.json")
	if err := SaveModel(model, path); err != nil {
		t.Fatalf("could not save: %v", err)
	}

	reloaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("could not load: %v", err)
	}

	for i, block := range reloaded.blocks {
		if block.attention.bq == nil || block.attention.bo == nil {
			t.Errorf("block %d lost its attention biases in the round trip", i)
		}
		if block.ffn.b1 == nil || block.ffn.b2 == nil {
			t.Errorf("block %d lost its mlp biases in the round trip", i)
		}
		if !block.ffn.useTanhGELU {
			t.Errorf("block %d came back using the wrong GELU", i)
		}
	}

	if !reloaded.IsLMHeadTied() {
		t.Errorf("the tied lm head flag did not survive the round trip")
	}

	after, err := reloaded.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// and the real check: identical output, not merely a plausible model
	for i := range before.data {
		diff := before.data[i] - after.data[i]
		if diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("logit %d changed across the round trip: %f vs %f", i, before.data[i], after.data[i])
		}
	}
}

func TestOldModelFilesStillLoadWithoutBiases(t *testing.T) {
	// a model trained before any of this existed has no bias fields in its
	// json at all. it has to come back exactly as it used to -> nil biases and
	// the erf GELU
	model, err := NewGPTModel(testVocabSize, testDModel, testNumLayers, testNumHeads, testMaxSeqLen, 0.02)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	path := filepath.Join(t.TempDir(), "legacy.json")
	if err := SaveModel(model, path); err != nil {
		t.Fatalf("could not save: %v", err)
	}

	// omitempty should have kept every optional field out of the file
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("could not read back the file: %v", err)
	}
	for _, unwanted := range []string{"\"bq\"", "\"b1\"", "use_tanh_gelu", "lm_head_tied"} {
		if strings.Contains(string(raw), unwanted) {
			t.Errorf("a model with no biases still wrote %s into its json", unwanted)
		}
	}

	reloaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("could not load: %v", err)
	}

	for i, block := range reloaded.blocks {
		if block.attention.bq != nil || block.ffn.b1 != nil {
			t.Errorf("block %d came back with biases it never had", i)
		}
		if block.ffn.useTanhGELU {
			t.Errorf("block %d came back on the tanh GELU, it was trained on the erf one", i)
		}
	}
	if reloaded.IsLMHeadTied() {
		t.Errorf("an untied model came back claiming a tied head")
	}
}

func TestCountParametersExcludesTheTiedHead(t *testing.T) {
	path := buildSyntheticGPT2(t, "")

	model, err := LoadGPT2(path, GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// work out what the count should be by hand, the same way the published
	// gpt-2 numbers are arrived at: the tied head is counted once, as wte
	expected := testVocabSize*testDModel + testMaxSeqLen*testDModel // embeddings
	expected += 2 * testDModel                                      // final layernorm
	perBlock := 4 * testDModel * testDModel                         // wq wk wv wo
	perBlock += 4 * testDModel                                      // their biases
	perBlock += 2 * testDModel * testDInner                         // w1 w2
	perBlock += testDInner + testDModel                             // their biases
	perBlock += 4 * testDModel                                      // two layernorms
	expected += testNumLayers * perBlock

	if got := model.CountParameters(); got != expected {
		t.Errorf("expected %d parameters, got %d", expected, got)
	}
}
