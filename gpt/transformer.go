package gpt

type FeedForward struct {
	dModel int
	dInner int
	w1     *Matrix
	w2     *Matrix
}

func NewFeedForward(dModel int, scale float32) *FeedForward {
	dInner := 4 * dModel
	w1 := NewMatrix(dModel, dInner)
	w2 := NewMatrix(dInner, dModel)
	w1.Randomize(scale)
	w2.Randomize(scale)
	return &FeedForward{
		dModel: dModel,
		dInner: dInner,
		w1:     w1,
		w2:     w2,
	}
}

func (ff *FeedForward) Forward(input *Matrix) (*Matrix, error) {
	hidden, err := input.MatrixMul(ff.w1)
	if err != nil {
		return nil, err
	}
	activated := hidden.GELU()
	output, err := activated.MatrixMul(ff.w2)
	if err != nil {
		return nil, err
	}
	return output, nil
}

type TransformerBlock struct {
	attention         *CausalAttention
	ffn               *FeedForward
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

func (tb *TransformerBlock) Forward(input *Matrix) (*Matrix, error) {
	norm1 := input.LayerNorm(tb.ln1Gamma, tb.ln1Beta, tb.epsilon)
	attnOut, err := tb.attention.Forward(norm1)
	if err != nil {
		return nil, err
	}
	res1, err := input.MatrixAdd(attnOut)
	if err != nil {
		return nil, err
	}
	norm2 := res1.LayerNorm(tb.ln2Gamma, tb.ln2Beta, tb.epsilon)
	ffnOut, err := tb.ffn.Forward(norm2)
	if err != nil {
		return nil, err
	}
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

	ffnGELUOut := ffnW1Out.GELU()

	ffnOut, err := ffnGELUOut.MatrixMul(tb.ffn.w2)
	if err != nil {
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
	}

	return output, acts, nil
}

func (tb *TransformerBlock) Backward(acts *BlockActivations, dOut *Matrix) (*Matrix, error) {
	// output = res1 + ffnOut → d_ffnOut = dOut, d_res1 gets dOut (first contribution)

	dFfnGELUOut, dW2, err := acts.ffnGELUOut.MatrixMulBackward(tb.ffn.w2, dOut)
	if err != nil {
		return nil, err
	}
	tb.ffn.w2.AccumulateGrad(dW2)

	dFfnW1Out := GELUBackward(acts.ffnW1Out, dFfnGELUOut)

	dNorm2, dW1, err := acts.norm2Out.MatrixMulBackward(tb.ffn.w1, dFfnW1Out)
	if err != nil {
		return nil, err
	}
	tb.ffn.w1.AccumulateGrad(dW1)

	dRes1FromLN2, dLn2Gamma, dLn2Beta := LayerNormBackward(acts.res1, tb.ln2Gamma, tb.epsilon, dNorm2)
	tb.ln2Gamma.AccumulateGrad(dLn2Gamma)
	tb.ln2Beta.AccumulateGrad(dLn2Beta)

	// total d_res1 = dOut (from residual) + dRes1FromLN2 (from LN2 path)
	totalDRes1, err := dOut.MatrixAdd(dRes1FromLN2)
	if err != nil {
		return nil, err
	}

	// res1 = blockInput + attnOut → both receive totalDRes1
	dNorm1, err := tb.attention.Backward(acts.attnActs, totalDRes1)
	if err != nil {
		return nil, err
	}

	dBlockInputFromLN1, dLn1Gamma, dLn1Beta := LayerNormBackward(acts.blockInput, tb.ln1Gamma, tb.epsilon, dNorm1)
	tb.ln1Gamma.AccumulateGrad(dLn1Gamma)
	tb.ln1Beta.AccumulateGrad(dLn1Beta)

	// total d_blockInput = totalDRes1 (from res1 skip) + dBlockInputFromLN1 (from LN1 path)
	dBlockInput, err := totalDRes1.MatrixAdd(dBlockInputFromLN1)
	if err != nil {
		return nil, err
	}

	return dBlockInput, nil
}

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
