package gpt

import (
	"fmt"
	"strings"
	"testing"
)

// buildReferenceFromModel runs the model and writes its own activations out as
// a reference dump.
//
// yes, this is circular as a correctness check -> it cannot tell us the GELU is
// right, only the real pytorch dump can do that. what it DOES test is all the
// machinery around the comparison: the metadata round trip, the shape checks,
// the diff maths and the pass verdict. if this is broken then the real dump
// would give us garbage verdicts and we would not know which to trust
func buildReferenceFromModel(t *testing.T, model *GPTModel, tokenIDs []int, tokenMetadata string) string {
	t.Helper()

	order, points, err := CollectActivations(model, tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error collecting activations: %v", err)
	}

	tensors := map[string][]float32{}
	shapes := map[string][]int{}

	for _, name := range order {
		m := points[name]
		values := make([]float32, len(m.data))
		copy(values, m.data)
		tensors[name] = values
		shapes[name] = []int{m.rows, m.cols}
	}

	return writeTestSafeTensors(t, tensors, shapes, map[string]string{
		"tokens": tokenMetadata,
		"prompt": "synthetic",
	})
}

func TestVerifyAgainstReferencePasses(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3}
	refPath := buildReferenceFromModel(t, model, tokenIDs, "1,2,3")

	diffs, gotTokens, allPassed, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !allPassed {
		for _, d := range diffs {
			if !d.Passed(AbsoluteTolerance(1e-6)) {
				t.Errorf("%s diverged: status %s, max diff %g", d.Name, d.Status, d.MaxAbsDiff)
			}
		}
		t.Fatalf("a model compared against its own activations should pass")
	}

	if len(gotTokens) != len(tokenIDs) {
		t.Errorf("expected %d tokens back, got %d", len(tokenIDs), len(gotTokens))
	}
	for i, id := range tokenIDs {
		if gotTokens[i] != id {
			t.Errorf("token %d: expected %d, got %d", i, id, gotTokens[i])
		}
	}

	// every point should have been genuinely compared, not skipped
	for _, d := range diffs {
		if d.Status != "ok" {
			t.Errorf("%s came back as %q, expected it to be compared", d.Name, d.Status)
		}
	}
}

func TestVerifyAgainstReferenceCatchesADivergence(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3}

	// take a correct dump, then nudge one value in the middle of the stack.
	// this is the shape of a real bug -> one layer wrong, everything after it
	// wrong as a consequence
	order, points, err := CollectActivations(model, tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tensors := map[string][]float32{}
	shapes := map[string][]int{}
	for _, name := range order {
		m := points[name]
		values := make([]float32, len(m.data))
		copy(values, m.data)
		if name == "h.1.ln_1" {
			values[0] += 0.25
		}
		tensors[name] = values
		shapes[name] = []int{m.rows, m.cols}
	}

	refPath := writeTestSafeTensors(t, tensors, shapes, map[string]string{"tokens": "1,2,3"})

	diffs, _, allPassed, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if allPassed {
		t.Fatalf("a tampered reference should not pass")
	}

	// and the failure has to be reported against the right tensor, otherwise
	// the whole "first divergence" idea is useless
	var firstFailure string
	for _, d := range diffs {
		if !d.Passed(AbsoluteTolerance(1e-6)) {
			firstFailure = d.Name
			break
		}
	}

	if firstFailure != "h.1.ln_1" {
		t.Errorf("expected the first divergence at h.1.ln_1, got %q", firstFailure)
	}
}

func TestVerifyNeedsTokenMetadata(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// a dump with no token list is useless to us -> we would have no idea what
	// prompt to run, and silently picking one would compare two different
	// things and call them different
	refPath := writeTestSafeTensors(t,
		map[string][]float32{"embed": {1, 2, 3, 4}},
		map[string][]int{"embed": {1, 4}},
		nil,
	)

	if _, _, _, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6)); err == nil {
		t.Errorf("expected an error when the reference carries no token metadata")
	}
}

func TestVerifySkipsTensorsTheReferenceDoesNotHave(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// a partial dump is legitimate -> you might only care about the logits.
	// those points should be reported as skipped rather than failed
	refPath := writeTestSafeTensors(t,
		map[string][]float32{"tokens_placeholder": {0}},
		map[string][]int{"tokens_placeholder": {1}},
		map[string]string{"tokens": "1,2,3"},
	)

	diffs, _, allPassed, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !allPassed {
		t.Errorf("a dump with nothing to compare should not count as a failure")
	}

	for _, d := range diffs {
		if d.Status != "missing" {
			t.Errorf("%s came back as %q, expected missing", d.Name, d.Status)
		}
	}
}

func TestVerifyRejectsAReferenceFromAWiderModel(t *testing.T) {
	// the bug that motivated this check: gpt2-medium weights (1024 wide)
	// paired with a reference dumped from gpt2 (768 wide). every tensor
	// mismatched and it read like a transpose bug
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// a reference claiming a hidden dim of 8 where our model is 4 wide
	wrongWidth := testDModel * 2
	refPath := writeTestSafeTensors(t,
		map[string][]float32{
			"ln_f":    make([]float32, 3*wrongWidth),
			"h.0.out": make([]float32, 3*wrongWidth),
			"h.1.out": make([]float32, 3*wrongWidth),
		},
		map[string][]int{
			"ln_f":    {3, wrongWidth},
			"h.0.out": {3, wrongWidth},
			"h.1.out": {3, wrongWidth},
		},
		map[string]string{"tokens": "1,2,3", "model": "something-else"},
	)

	_, _, _, err = VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err == nil {
		t.Fatalf("expected a mismatched reference to be refused outright")
	}

	// the message has to name the real problem, not hand back a table
	for _, want := range []string{"different models", "something-else"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should mention %q, got: %v", want, err)
		}
	}
}

func TestVerifyRejectsAReferenceWithADifferentLayerCount(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// correct width, but one block too many -> a different model depth
	tensors := map[string][]float32{"ln_f": make([]float32, 3*testDModel)}
	shapes := map[string][]int{"ln_f": {3, testDModel}}
	for i := 0; i < testNumLayers+1; i++ {
		name := fmt.Sprintf("h.%d.out", i)
		tensors[name] = make([]float32, 3*testDModel)
		shapes[name] = []int{3, testDModel}
	}

	refPath := writeTestSafeTensors(t, tensors, shapes, map[string]string{"tokens": "1,2,3"})

	if _, _, _, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6)); err == nil {
		t.Errorf("expected a reference with a different layer count to be refused")
	}
}

func TestVerifyAcceptsAPartialReference(t *testing.T) {
	// a dump that only carries the logits has no ln_f and no h.N.out, so the
	// cross check cannot determine its shape. that must not be treated as a
	// mismatch -> partial dumps are legitimate
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	order, points, err := CollectActivations(model, []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = order

	logits := points["logits"]
	values := make([]float32, len(logits.data))
	copy(values, logits.data)

	refPath := writeTestSafeTensors(t,
		map[string][]float32{"logits": values},
		map[string][]int{"logits": {logits.rows, logits.cols}},
		map[string]string{"tokens": "1,2,3"},
	)

	diffs, _, allPassed, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err != nil {
		t.Fatalf("a partial reference should be accepted, got %v", err)
	}
	if !allPassed {
		t.Errorf("a partial reference that matches should pass")
	}

	// and the one tensor it did carry should have been genuinely compared
	var compared int
	for _, d := range diffs {
		if d.Status == "ok" {
			compared++
		}
	}
	if compared != 1 {
		t.Errorf("expected exactly 1 compared tensor, got %d", compared)
	}
}

func TestVerifyUsesShapeMetadataWhenPresent(t *testing.T) {
	// a dump that states its own shape should be cross checked on that, even
	// when it carries no tensors we could infer the shape from. this is what
	// makes a logits-only dump from the wrong model still get caught
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	refPath := writeTestSafeTensors(t,
		map[string][]float32{"logits": make([]float32, 3*testVocabSize)},
		map[string][]int{"logits": {3, testVocabSize}},
		map[string]string{
			"tokens":  "1,2,3",
			"model":   "gpt2-medium",
			"n_layer": "24",
			"n_embd":  "1024",
		},
	)

	_, _, _, err = VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6))
	if err == nil {
		t.Fatalf("expected the metadata shape to be caught as a mismatch")
	}
	if !strings.Contains(err.Error(), "24 layers") {
		t.Errorf("the error should quote the reference's stated depth, got: %v", err)
	}
}

func TestVerifyIgnoresJunkShapeMetadata(t *testing.T) {
	// unparseable metadata must not be treated as a mismatch -> we fall back
	// to reading the shape off the tensors instead
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3}
	order, points, err := CollectActivations(model, tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tensors := map[string][]float32{}
	shapes := map[string][]int{}
	for _, name := range order {
		m := points[name]
		values := make([]float32, len(m.data))
		copy(values, m.data)
		tensors[name] = values
		shapes[name] = []int{m.rows, m.cols}
	}

	refPath := writeTestSafeTensors(t, tensors, shapes, map[string]string{
		"tokens":  "1,2,3",
		"n_layer": "not a number",
		"n_embd":  "",
	})

	if _, _, allPassed, err := VerifyAgainstReference(model, refPath, AbsoluteTolerance(1e-6)); err != nil || !allPassed {
		t.Errorf("junk metadata should be ignored, got allPassed=%v err=%v", allPassed, err)
	}
}

func TestToleranceAcceptsFloat32NoiseOnLargeActivations(t *testing.T) {
	// these are the real numbers from a gpt2-medium run. block 3's mlp output
	// reaches a magnitude of 3308 because of gpt-2's outlier features, and at
	// that scale float32 cannot do better than about the eighth digit.
	// a flat absolute tolerance called this a bug, and it cost a round of
	// debugging something that was never wrong
	diff := TensorDiff{
		Name:        "h.3.mlp",
		Status:      "ok",
		MaxAbsDiff:  7.080e-3,
		MeanAbsDiff: 2.686e-6,
		RefMaxAbs:   3308.39,
		RefMeanAbs:  1.0276,
		MaxRelDiff:  7.080e-3 / 3308.39,
		MeanRelDiff: 2.686e-6 / 1.0276,
	}

	if !diff.Passed(DefaultTolerance()) {
		t.Errorf("float32 noise on a tensor of magnitude 3308 should pass: max rel %.2e, mean rel %.2e",
			diff.MaxRelDiff, diff.MeanRelDiff)
	}

	// and it must still fail a strict absolute check, which is what the unit
	// tests use when comparing a model against its own activations
	if diff.Passed(AbsoluteTolerance(1e-6)) {
		t.Errorf("a 7e-3 absolute difference should not pass an absolute tolerance of 1e-6")
	}
}

func TestToleranceRejectsASystematicError(t *testing.T) {
	// what a real bug looks like: the error is spread across every element
	// rather than sitting in a few big ones. swapping gpt-2's tanh GELU for
	// the erf form moves the mean relative error to roughly 1e-3, which is
	// about one hundred times the noise floor we actually measured
	diff := TensorDiff{
		Name:        "h.3.mlp",
		Status:      "ok",
		MaxAbsDiff:  1.0,
		MeanAbsDiff: 1.1e-3,
		RefMaxAbs:   3308.39,
		RefMeanAbs:  1.0276,
		MaxRelDiff:  1.0 / 3308.39,   // 3.0e-4, comfortably inside MaxRel
		MeanRelDiff: 1.1e-3 / 1.0276, // 1.1e-3, eleven times over MeanRel
	}

	if diff.Passed(DefaultTolerance()) {
		t.Errorf("an error smeared across every element should fail: mean rel %.2e", diff.MeanRelDiff)
	}

	// note the max relative error alone would have let this through, which is
	// exactly why both figures have to hold
	if diff.MaxRelDiff > DefaultTolerance().MaxRel {
		t.Errorf("this fixture is meant to pass the max check and fail on the mean, got max rel %.2e", diff.MaxRelDiff)
	}
}

func TestToleranceHandlesAnAllZeroReference(t *testing.T) {
	// a reference tensor of all zeros would divide by zero. the verifier leaves
	// the relative figures at zero in that case, so an exact match still
	// passes rather than coming back as NaN
	diff := TensorDiff{
		Name:        "zeros",
		Status:      "ok",
		MaxAbsDiff:  0,
		MeanAbsDiff: 0,
		RefMaxAbs:   0,
		RefMeanAbs:  0,
	}

	if !diff.Passed(DefaultTolerance()) {
		t.Errorf("an exact match against an all-zero reference should pass")
	}
}

func TestTensorScale(t *testing.T) {
	m, err := NewMatrixFromData(2, 3, []float32{1, -5, 2, 0, 3, -1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	maxAbs, meanAbs := tensorScale(m)

	if maxAbs != 5 {
		t.Errorf("expected max absolute value 5, got %f", maxAbs)
	}
	// (1 + 5 + 2 + 0 + 3 + 1) / 6 = 2
	if meanAbs != 2 {
		t.Errorf("expected mean absolute value 2, got %f", meanAbs)
	}
}

func TestVerifyReportsScaleAlongsideTheDiff(t *testing.T) {
	// the scale columns are what make the absolute numbers interpretable, so
	// they have to actually be populated
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	refPath := buildReferenceFromModel(t, model, []int{1, 2, 3}, "1,2,3")

	diffs, _, _, err := VerifyAgainstReference(model, refPath, DefaultTolerance())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, d := range diffs {
		if d.Status != "ok" {
			continue
		}
		if d.RefMaxAbs == 0 && d.RefMeanAbs == 0 {
			// logits and activations in this synthetic model are never all
			// zero, so this would mean we simply forgot to measure the scale
			t.Errorf("%s came back with no scale measured", d.Name)
		}
	}
}

func TestGPT2SizeName(t *testing.T) {
	cases := []struct {
		dModel, layers int
		expected       string
	}{
		{768, 12, "gpt2"},
		{1024, 24, "gpt2-medium"},
		{1280, 36, "gpt2-large"},
		{1600, 48, "gpt2-xl"},
		// the shape that caused the confusion -> medium's width with small's
		// depth is not a real model, so we should not claim it is one
		{1024, 12, ""},
		{4, 2, ""},
	}

	for _, c := range cases {
		if got := gpt2SizeName(c.dModel, c.layers); got != c.expected {
			t.Errorf("%d wide, %d layers: expected %q, got %q", c.dModel, c.layers, c.expected, got)
		}
	}
}

func TestParseReferenceTokensRejectsJunk(t *testing.T) {
	ref := &SafeTensors{Metadata: map[string]string{"tokens": "1,2,banana"}}

	if _, err := parseReferenceTokens(ref); err == nil {
		t.Errorf("expected an error for a non-numeric token id")
	}

	ref = &SafeTensors{Metadata: map[string]string{"tokens": "  "}}
	if _, err := parseReferenceTokens(ref); err == nil {
		t.Errorf("expected an error for an empty token list")
	}

	// whitespace around the numbers is fine though, since the python side
	// could reasonably write them either way
	ref = &SafeTensors{Metadata: map[string]string{"tokens": " 15496 , 11 , 616 "}}
	tokens, err := parseReferenceTokens(ref)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := []int{15496, 11, 616}
	for i, id := range expected {
		if tokens[i] != id {
			t.Errorf("token %d: expected %d, got %d", i, id, tokens[i])
		}
	}
}
