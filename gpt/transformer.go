package gpt

// Transformer Block Implementation
// In the case of the GPT architecture
// the transformer block combines attention,
// feed-foward networks (MLPs) and residual connections
// which end up looking like this:
// 1. attention:
//   - input -> layer normalization -> causal multi-head attention -> residual connection
//
// 2. feed-forward sub-block (MLP)
//   - output from attention -> layer normalization -> linear layer 1 -> GELU Activation
//     -> Lineral Layer 2 -> Residual Connection

type FeedForward struct {
	dModel int
	dInner int     // likely 4 x dModel
	w1     *Matrix // shape: dModel x dInner
	w2     *Matrix // shape: dInner x dModel

	// bias vectors -> b1 is 1 x dInner, b2 is 1 x dModel
	// nil for anything we train ourselves, filled in for pretrained gpt-2
	// (c_fc.bias and c_proj.bias respectively)
	b1 *Matrix
	b2 *Matrix

	// which flavour of GELU to run between the two projections.
	// false -> the exact erf form, which is what our own training uses
	// true  -> the tanh approximation, which is what gpt-2 was trained with
	// see the comments on GELUTanh for why this distinction matters
	useTanhGELU bool
}

func NewFeedForward(dModel int, scale float32) *FeedForward {
	dInner := 4 * dModel
	w1 := NewMatrix(dModel, dInner)
	w2 := NewMatrix(dInner, dModel)

	// weight randomization
	w1.Randomize(scale)
	w2.Randomize(scale)

	return &FeedForward{
		dModel: dModel,
		dInner: dInner,
		w1:     w1,
		w2:     w2,
	}
}

// activate runs whichever GELU this FFN was built for
func (ff *FeedForward) activate(input *Matrix) *Matrix {
	if ff.useTanhGELU {
		return input.GELUTanh()
	}
	return input.GELU()
}

// activateBackward is the matching derivative, picked the same way.
// if these two ever disagree the gradients are silently wrong, which is why
// they both read the one flag instead of being chosen at the call site
func (ff *FeedForward) activateBackward(input *Matrix, dOut *Matrix) *Matrix {
	if ff.useTanhGELU {
		return GELUTanhBackward(input, dOut)
	}
	return GELUBackward(input, dOut)
}

// forward executes the FFN: linear1-> GELU -> linear2
func (ff *FeedForward) Forward(input *Matrix) (*Matrix, error) {
	// first linear projection aka expansion
	hidden, err := input.MatrixMul(ff.w1)
	if err != nil {
		return nil, err
	}
	if err := hidden.AddRowBias(ff.b1); err != nil {
		return nil, err
	}

	// apply gely activation
	activated := ff.activate(hidden)

	// second linear projection which contracts back to dModel
	output, err := activated.MatrixMul(ff.w2)
	if err != nil {
		return nil, err
	}
	if err := output.AddRowBias(ff.b2); err != nil {
		return nil, err
	}

	return output, nil
}

// transformer Block
// combining Attention + FFN + Residuals
type TransformerBlock struct {
	attention *CausalAttention
	ffn       *FeedForward

	// layerNorm parameters for normalization sub-layers
	// we can keep them nil initially
	ln1Gamma, ln1Beta *Matrix
	ln2Gamma, ln2Beta *Matrix
	epsilon           float32
}

func NewTransformerBlock(dModel, numHeads int, scale float32) (*TransformerBlock, error) {
	attn, err := NewCausalAttention(dModel, numHeads, scale)
	if err != nil {
		return nil, err
	}

	ffn := NewFeedForward(dModel, scale)

	ln1Gamma := NewMatrix(1, dModel)
	ln1Beta := NewMatrix(1, dModel)
	ln2Gamma := NewMatrix(1, dModel)
	ln2Beta := NewMatrix(1, dModel)
	for j := 0; j < dModel; j++ {
		ln1Gamma.data[j] = 1.0
		ln2Gamma.data[j] = 1.0
	}

	return &TransformerBlock{
		attention: attn,
		ffn:       ffn,
		ln1Gamma:  ln1Gamma,
		ln1Beta:   ln1Beta,
		ln2Gamma:  ln2Gamma,
		ln2Beta:   ln2Beta,
		epsilon:   1e-5,
	}, nil
}

// forward passes the input tensor through the complete transformer block
func (tb *TransformerBlock) Forward(input *Matrix) (*Matrix, error) {
	// pre-layer norm
	norm1 := input.LayerNorm(tb.ln1Gamma, tb.ln1Beta, tb.epsilon)

	// run the causal multi-head attention
	attnOut, err := tb.attention.Forward(norm1)
	if err != nil {
		return nil, err
	}

	// add the original input to attention out -> residual connection
	res1, err := input.MatrixAdd(attnOut)
	if err != nil {
		return nil, err
	}

	// pre-player normalize result before sending to ffn
	norm2 := res1.LayerNorm(tb.ln2Gamma, tb.ln2Beta, tb.epsilon)

	// now we run the feedback forward network
	ffnOut, err := tb.ffn.Forward(norm2)
	if err != nil {
		return nil, err
	}

	// finally, the residual connection, adding res1 to ffn output
	output, err := res1.MatrixAdd(ffnOut)
	if err != nil {
		return nil, err
	}

	return output, nil
}

type BlockActivations struct {
	blockInput *Matrix
	norm1Out   *Matrix
	attnActs   *AttentionActivations
	attnOut    *Matrix
	res1       *Matrix
	norm2Out   *Matrix
	ffnW1Out   *Matrix
	ffnGELUOut *Matrix

	// the backward pass doesn't need these two, but the reference comparison
	// in verify.go does -> they're the points a pytorch gpt-2 lets us hook
	ffnOut *Matrix
	output *Matrix
}

func (tb *TransformerBlock) ForwardSave(input *Matrix) (*Matrix, *BlockActivations, error) {
	norm1 := input.LayerNorm(tb.ln1Gamma, tb.ln1Beta, tb.epsilon)

	attnOut, attnActs, err := tb.attention.ForwardSave(norm1)
	if err != nil {
		return nil, nil, err
	}

	res1, err := input.MatrixAdd(attnOut)
	if err != nil {
		return nil, nil, err
	}

	norm2 := res1.LayerNorm(tb.ln2Gamma, tb.ln2Beta, tb.epsilon)

	ffnW1Out, err := norm2.MatrixMul(tb.ffn.w1)
	if err != nil {
		return nil, nil, err
	}
	if err := ffnW1Out.AddRowBias(tb.ffn.b1); err != nil {
		return nil, nil, err
	}

	ffnGELUOut := tb.ffn.activate(ffnW1Out)

	ffnOut, err := ffnGELUOut.MatrixMul(tb.ffn.w2)
	if err != nil {
		return nil, nil, err
	}
	if err := ffnOut.AddRowBias(tb.ffn.b2); err != nil {
		return nil, nil, err
	}

	output, err := res1.MatrixAdd(ffnOut)
	if err != nil {
		return nil, nil, err
	}

	acts := &BlockActivations{
		blockInput: input,
		norm1Out:   norm1,
		attnActs:   attnActs,
		attnOut:    attnOut,
		res1:       res1,
		norm2Out:   norm2,
		ffnW1Out:   ffnW1Out,
		ffnGELUOut: ffnGELUOut,
		ffnOut:     ffnOut,
		output:     output,
	}

	return output, acts, nil
}

func (tb *TransformerBlock) Backward(acts *BlockActivations, dOut *Matrix) (*Matrix, error) {
	dFfnGELUOut, dW2, err := acts.ffnGELUOut.MatrixMulBackward(tb.ffn.w2, dOut)
	if err != nil {
		return nil, err
	}
	tb.ffn.w2.AccumulateGrad(dW2)
	accumulateBiasGrad(tb.ffn.b2, dOut)

	dFfnW1Out := tb.ffn.activateBackward(acts.ffnW1Out, dFfnGELUOut)

	dNorm2, dW1, err := acts.norm2Out.MatrixMulBackward(tb.ffn.w1, dFfnW1Out)
	if err != nil {
		return nil, err
	}
	tb.ffn.w1.AccumulateGrad(dW1)
	accumulateBiasGrad(tb.ffn.b1, dFfnW1Out)

	dRes1FromLN2, dLn2Gamma, dLn2Beta := LayerNormBackward(acts.res1, tb.ln2Gamma, tb.epsilon, dNorm2)
	tb.ln2Gamma.AccumulateGrad(dLn2Gamma)
	tb.ln2Beta.AccumulateGrad(dLn2Beta)

	totalDRes1, err := dOut.MatrixAdd(dRes1FromLN2)
	if err != nil {
		return nil, err
	}

	dNorm1, err := tb.attention.Backward(acts.attnActs, totalDRes1)
	if err != nil {
		return nil, err
	}

	dBlockInputFromLN1, dLn1Gamma, dLn1Beta := LayerNormBackward(acts.blockInput, tb.ln1Gamma, tb.epsilon, dNorm1)
	tb.ln1Gamma.AccumulateGrad(dLn1Gamma)
	tb.ln1Beta.AccumulateGrad(dLn1Beta)

	dBlockInput, err := totalDRes1.MatrixAdd(dBlockInputFromLN1)
	if err != nil {
		return nil, err
	}

	return dBlockInput, nil
}

// identical to the one above, except using cache here
func (tb *TransformerBlock) ForwardWithCache(input *Matrix, cache *KvCache) (*Matrix, error) {
	normed := input.LayerNorm(tb.ln1Gamma, tb.ln1Beta, tb.epsilon)
	attnOut, err := tb.attention.ForwardWithCache(normed, cache)
	if err != nil {
		return nil, err
	}
	res1, err := input.MatrixAdd(attnOut)
	if err != nil {
		return nil, err
	}
	normedFFN := res1.LayerNorm(tb.ln2Gamma, tb.ln2Beta, tb.epsilon)
	ffnOut, err := tb.ffn.Forward(normedFFN)
	if err != nil {
		return nil, err
	}
	output, err := res1.MatrixAdd(ffnOut)
	if err != nil {
		return nil, err
	}
	return output, nil
}
