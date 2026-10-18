package gpt

import (
	"fmt"
)

// GPT-2 Checkpoint Loader
// this is the bridge between a real pretrained gpt-2 and our own model struct.
//
// the build architecture is literally GPT-2, therefore, for simple training data
// we can train our own model, but training something like GPT from scratch
// using this arch on a CPU is a nightmare. it is doable, but in roughly 50k years
// as a result, i wanted the option to load pre-trained weights and use them to do
// inference with the current tech we got. so here's the deltas we had to deal with here:
//
// 1. bias terms       -> we had none, gpt-2 has four per block (see matrix.go)
// 2. the GELU flavour -> gpt-2 wants the tanh approximation (see activations.go)
// 3. a fused c_attn   -> gpt-2 keeps Q, K and V in one matrix, we keep three
// 4. the lm head      -> gpt-2 ties it to wte, which is stored transposed
// 5. the file format  -> safetensors instead of our json (see safetensors.go)
//
// everything else lines up byte for byte, no transposing, because huggingface
// gpt-2 is written with Conv1D (weight shaped in x out, forward x @ w + b)
// rather than nn.Linear.
//
// quick fyi: if any of the five points above is wrong,
// this loader will NOT error. you will get a model that produces fluent,
// grammatical nonsense, which looks exactly like an undertrained model. that is
// what verify.go exists for -> load, then compare against reference
// activations before you trust a single token of output

type GPT2Config struct {
	// NumHeads can't be derived from the weights -> nothing in the file says
	// how the hidden dimension is sliced up. leave it at 0 and we infer it
	// from dModel using the published gpt-2 sizes
	NumHeads int

	// Prefix in front of every tensor name. a checkpoint saved from
	// GPT2LMHeadModel prefixes everything with "transformer.", one saved from
	// GPT2Model does not. leave it empty and we sniff it out
	Prefix string

	// Quantize converts each matmul weight to int8 as it is read, rather than
	// after the whole model is in memory.
	//
	// that ordering is the entire point. quantizing afterwards works but does
	// nothing for PEAK memory, because for a moment you hold the float32
	// weights and their int8 copies at the same time -> measured on
	// gpt2-medium, post-hoc quantization actually pushed the peak UP, from
	// 1863MB to 2169MB. doing it per tensor during the load means only one
	// float32 tensor is ever live at a time
	Quantize bool
}

// the four published gpt-2 sizes, keyed by hidden dimension.
// small 768/12 heads, medium 1024/16, large 1280/20, xl 1600/25
var gpt2HeadsByDModel = map[int]int{
	768:  12,
	1024: 16,
	1280: 20,
	1600: 25,
}

// and the same four keyed by hidden dim plus layer count, purely so error
// messages can say "looks like gpt2-medium" instead of making you look up
// which size has 24 layers
var gpt2Names = map[[2]int]string{
	{768, 12}:  "gpt2",
	{1024, 24}: "gpt2-medium",
	{1280, 36}: "gpt2-large",
	{1600, 48}: "gpt2-xl",
}

// gpt2SizeName names a model from its shape, or returns empty for anything
// that is not one of the published sizes
func gpt2SizeName(dModel, numLayers int) string {
	return gpt2Names[[2]int{dModel, numLayers}]
}

// LoadGPT2 reads a huggingface gpt-2 safetensors checkpoint and wires it into
// a GPTModel we can run our normal forward pass on.
//
// the shape of the model (vocab, hidden dim, layer count, context length) is
// read off the weights themselves rather than trusted from a config file, so
// this works for any of the gpt-2 sizes without being told which one it is
func LoadGPT2(path string, cfg GPT2Config) (*GPTModel, error) {
	st, err := OpenSafeTensors(path)
	if err != nil {
		return nil, err
	}
	// the file buffer is big -> drop it as soon as we are done converting
	defer st.Close()

	prefix, err := detectGPT2Prefix(st, cfg.Prefix)
	if err != nil {
		return nil, err
	}

	// the embeddings tell us almost everything about the model's shape:
	// wte is vocabSize x dModel, wpe is maxSeqLen x dModel
	tokenEmbedWeights, err := st.Matrix(prefix + "wte.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: could not read the token embeddings: %v", err)
	}

	posEmbedWeights, err := st.Matrix(prefix + "wpe.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: could not read the positional embeddings: %v", err)
	}

	vocabSize := tokenEmbedWeights.rows
	dModel := tokenEmbedWeights.cols
	maxSeqLen := posEmbedWeights.rows

	if posEmbedWeights.cols != dModel {
		return nil, fmt.Errorf("gpt2: wte says dModel is %d but wpe says %d", dModel, posEmbedWeights.cols)
	}

	// the layer count is however many h.N. blocks are actually in the file
	numLayers := countGPT2Layers(st, prefix)
	if numLayers == 0 {
		return nil, fmt.Errorf("gpt2: found no transformer blocks under prefix %q, is this really a gpt-2 checkpoint?", prefix)
	}

	numHeads := cfg.NumHeads
	if numHeads == 0 {
		known, ok := gpt2HeadsByDModel[dModel]
		if !ok {
			return nil, fmt.Errorf("gpt2: dModel %d is not one of the known gpt-2 sizes, pass NumHeads explicitly", dModel)
		}
		numHeads = known
	}
	if dModel%numHeads != 0 {
		return nil, fmt.Errorf("gpt2: dModel (%d) is not divisible by numHeads (%d)", dModel, numHeads)
	}

	model := &GPTModel{
		vocabSize: vocabSize,
		dModel:    dModel,
		numLayers: numLayers,
		numHeads:  numHeads,
		maxSeqLen: maxSeqLen,
		epsilon:   1e-5, // what gpt-2 was trained with, same as our default
		blocks:    make([]*TransformerBlock, numLayers),
	}

	model.tokenEmbed = &Embedding{
		vocabSize: vocabSize,
		hiddenDim: dModel,
		weights:   tokenEmbedWeights,
	}

	model.positionEmbed = &PositionalEmbedding{
		maxSeqLen: maxSeqLen,
		hiddenDim: dModel,
		weights:   posEmbedWeights,
	}

	for i := 0; i < numLayers; i++ {
		block, err := loadGPT2Block(st, prefix, i, dModel, numHeads, cfg.Quantize)
		if err != nil {
			return nil, err
		}
		model.blocks[i] = block
	}

	// the final layernorm before the lm head
	model.finalLnGamma, err = st.Matrix(prefix + "ln_f.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: %v", err)
	}
	model.finalLnBeta, err = st.Matrix(prefix + "ln_f.bias")
	if err != nil {
		return nil, fmt.Errorf("gpt2: %v", err)
	}

	lmHead, tied, err := loadGPT2LMHead(st, prefix, tokenEmbedWeights, dModel, vocabSize)
	if err != nil {
		return nil, err
	}
	model.lmHead = lmHead
	model.lmHeadTied = tied

	if cfg.Quantize {
		// the lm head is the single biggest matmul weight -> 50257 x dModel,
		// so 206MB of float32 for medium and 322MB for xl
		if err := model.lmHead.Quantize(); err != nil {
			return nil, fmt.Errorf("gpt2 lm head: %v", err)
		}
	}

	return model, nil
}

// detectGPT2Prefix figures out whether tensor names carry a "transformer."
// prefix or not. we test for wte.weight because every gpt-2 has exactly one
func detectGPT2Prefix(st *SafeTensors, requested string) (string, error) {
	// an explicit prefix is taken at face value, but we still sanity check it
	// so a typo fails here instead of twenty tensors later
	if requested != "" {
		if !st.Has(requested + "wte.weight") {
			return "", fmt.Errorf("gpt2: no tensor %q in this checkpoint", requested+"wte.weight")
		}
		return requested, nil
	}

	for _, candidate := range []string{"", "transformer."} {
		if st.Has(candidate + "wte.weight") {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("gpt2: could not find wte.weight with or without a transformer. prefix. tensors in this file: %v", st.Names())
}

// countGPT2Layers walks upwards from block 0 until a block is missing.
// counting forward like this rather than parsing out the highest index means a
// checkpoint with a gap in it fails loudly later instead of silently losing a
// layer in the middle of the stack
func countGPT2Layers(st *SafeTensors, prefix string) int {
	count := 0
	for {
		if !st.Has(fmt.Sprintf("%sh.%d.ln_1.weight", prefix, count)) {
			return count
		}
		count++
	}
}

// loadGPT2Block builds one transformer block out of the checkpoint
func loadGPT2Block(st *SafeTensors, prefix string, index, dModel, numHeads int, quantize bool) (*TransformerBlock, error) {
	base := fmt.Sprintf("%sh.%d.", prefix, index)

	// pull everything we need up front, so a missing tensor is reported with
	// its real name instead of blowing up as a nil dereference mid-forward
	names := []string{
		"ln_1.weight", "ln_1.bias",
		"attn.c_attn.weight", "attn.c_attn.bias",
		"attn.c_proj.weight", "attn.c_proj.bias",
		"ln_2.weight", "ln_2.bias",
		"mlp.c_fc.weight", "mlp.c_fc.bias",
		"mlp.c_proj.weight", "mlp.c_proj.bias",
	}

	tensors := make(map[string]*Matrix, len(names))
	for _, name := range names {
		m, err := st.Matrix(base + name)
		if err != nil {
			return nil, fmt.Errorf("gpt2 block %d: %v", index, err)
		}
		tensors[name] = m
	}

	// gpt-2 keeps Q, K and V fused into a single dModel x 3*dModel matrix,
	// concatenated along the OUTPUT dimension. we keep them separate, so we
	// slice it into thirds
	wq, wk, wv, err := splitFusedQKV(tensors["attn.c_attn.weight"], dModel)
	if err != nil {
		return nil, fmt.Errorf("gpt2 block %d: %v", index, err)
	}

	bq, bk, bv, err := splitFusedBias(tensors["attn.c_attn.bias"], dModel)
	if err != nil {
		return nil, fmt.Errorf("gpt2 block %d: %v", index, err)
	}

	attention := &CausalAttention{
		dModel:   dModel,
		numHeads: numHeads,
		headDim:  dModel / numHeads,
		wq:       wq,
		wk:       wk,
		wv:       wv,
		wo:       tensors["attn.c_proj.weight"],
		bq:       bq,
		bk:       bk,
		bv:       bv,
		bo:       tensors["attn.c_proj.bias"],
	}

	w1 := tensors["mlp.c_fc.weight"]
	ffn := &FeedForward{
		dModel: dModel,
		dInner: w1.cols,
		w1:     w1,
		w2:     tensors["mlp.c_proj.weight"],
		b1:     tensors["mlp.c_fc.bias"],
		b2:     tensors["mlp.c_proj.bias"],
		// the one line that keeps the activation honest
		useTanhGELU: true,
	}

	// quantize here, while only this block's float32 weights are live, rather
	// than once the whole model is loaded
	if quantize {
		for name, weight := range map[string]*Matrix{
			"wq": attention.wq, "wk": attention.wk,
			"wv": attention.wv, "wo": attention.wo,
			"w1": ffn.w1, "w2": ffn.w2,
		} {
			if err := weight.Quantize(); err != nil {
				return nil, fmt.Errorf("gpt2 block %d %s: %v", index, name, err)
			}
		}
	}

	return &TransformerBlock{
		attention: attention,
		ffn:       ffn,
		ln1Gamma:  tensors["ln_1.weight"],
		ln1Beta:   tensors["ln_1.bias"],
		ln2Gamma:  tensors["ln_2.weight"],
		ln2Beta:   tensors["ln_2.bias"],
		epsilon:   1e-5,
	}, nil
}

// splitFusedQKV cuts gpt-2's single c_attn weight into our three separate
// projections.
//
// the source is dModel x 3*dModel in row-major order, so every row holds
// [ q slice | k slice | v slice ], each dModel wide. that means we can't just
// take the first third of the buffer -> we have to walk row by row and pick
// the right window out of each one
func splitFusedQKV(fused *Matrix, dModel int) (*Matrix, *Matrix, *Matrix, error) {
	if fused.rows != dModel || fused.cols != 3*dModel {
		return nil, nil, nil, fmt.Errorf("c_attn.weight should be %dx%d but it is %dx%d", dModel, 3*dModel, fused.rows, fused.cols)
	}

	wq := NewMatrix(dModel, dModel)
	wk := NewMatrix(dModel, dModel)
	wv := NewMatrix(dModel, dModel)

	for r := 0; r < dModel; r++ {
		src := r * 3 * dModel
		dst := r * dModel
		copy(wq.data[dst:dst+dModel], fused.data[src:src+dModel])
		copy(wk.data[dst:dst+dModel], fused.data[src+dModel:src+2*dModel])
		copy(wv.data[dst:dst+dModel], fused.data[src+2*dModel:src+3*dModel])
	}

	return wq, wk, wv, nil
}

// splitFusedBias does the same thing for the c_attn bias, which is a single
// 1 x 3*dModel row. this one really is just three consecutive chunks
func splitFusedBias(fused *Matrix, dModel int) (*Matrix, *Matrix, *Matrix, error) {
	if fused.rows != 1 || fused.cols != 3*dModel {
		return nil, nil, nil, fmt.Errorf("c_attn.bias should be 1x%d but it is %dx%d", 3*dModel, fused.rows, fused.cols)
	}

	bq, err := NewMatrixFromData(1, dModel, fused.data[:dModel])
	if err != nil {
		return nil, nil, nil, err
	}
	bk, err := NewMatrixFromData(1, dModel, fused.data[dModel:2*dModel])
	if err != nil {
		return nil, nil, nil, err
	}
	bv, err := NewMatrixFromData(1, dModel, fused.data[2*dModel:3*dModel])
	if err != nil {
		return nil, nil, nil, err
	}

	return bq, bk, bv, nil
}

// loadGPT2LMHead produces the dModel x vocabSize projection our forward pass
// expects.
//
// this is the one place in the whole loader where a transpose is unavoidable.
// gpt-2 ties the output head to the token embeddings, and wte is stored as
// vocabSize x dModel so it can be used as a lookup table. we need the other
// orientation to do normed.MatrixMul(lmHead), so we flip it.
//
// most checkpoints don't ship an lm_head.weight at all because of that tying,
// but we check for one anyway and prefer it when it exists.
//
// the bool we return says whether we ended up tying, which the parameter count
// needs to know so it doesn't count these weights twice
func loadGPT2LMHead(st *SafeTensors, prefix string, tokenEmbedWeights *Matrix, dModel, vocabSize int) (*Matrix, bool, error) {
	// an untied head can live with or without the prefix depending on how the
	// checkpoint was saved, so try both
	for _, name := range []string{prefix + "lm_head.weight", "lm_head.weight"} {
		if !st.Has(name) {
			continue
		}

		head, err := st.Matrix(name)
		if err != nil {
			return nil, false, fmt.Errorf("gpt2: %v", err)
		}

		// stored the same way round as wte -> vocabSize x dModel
		if head.rows == vocabSize && head.cols == dModel {
			return head.Transpose(), false, nil
		}
		// already the orientation we want, take it as-is
		if head.rows == dModel && head.cols == vocabSize {
			return head, false, nil
		}

		return nil, false, fmt.Errorf("gpt2: %s has shape %dx%d, expected %dx%d or %dx%d", name, head.rows, head.cols, vocabSize, dModel, dModel, vocabSize)
	}

	// no explicit head, so it is tied to the embeddings
	return tokenEmbedWeights.Transpose(), true, nil
}
