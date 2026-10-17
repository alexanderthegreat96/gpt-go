package gpt

import (
	"fmt"
)

// Now it's time for the big gun
// the actual GPT Model
// a few stages to that:
// 1. token & positional embeddings -> turning token ids into dense vector reps combined with pos data
// 2. stack of transformer blocks -> multiple repeating transformer blocks
// 3. final layer normalization to stabilize the final hidden states before projection
// 4. language modeling head (lm head) -> a liner projection layer mapping dModel back into full
// the full vocab size to output raw logics for next-token prediction

type GPTModel struct {
	vocabSize int
	dModel    int
	numLayers int
	numHeads  int
	maxSeqLen int

	// the embeddings we use
	tokenEmbed    *Embedding
	positionEmbed *PositionalEmbedding

	// self-explanatory
	blocks []*TransformerBlock

	// normalization params
	finalLnGamma *Matrix
	finalLnBeta  *Matrix
	epsilon      float32

	// this is a weight matrix which maps the dModel back to vocabSize
	// with a shape of dModel x vocabSize
	// as a sidenote: in other implementations this can tie weights with tokenEmbed
	// for educational purposes a separate projection matrix is straight forward
	// will probably mess with this as well
	lmHead *Matrix
}

func NewGPTModel(vocabSize, dModel, numLayers, numHeads, maxSeqLen int, scale float32) (*GPTModel, error) {
	tokenEmbed := NewEmbedding(vocabSize, dModel, scale)
	positionEmbed := NewPositionalEmbedding(maxSeqLen, dModel, scale)

	blocks := make([]*TransformerBlock, numLayers)
	for i := 0; i < numLayers; i++ {
		block, err := NewTransformerBlock(dModel, numHeads, scale)
		if err != nil {
			return nil, err
		}
		blocks[i] = block
	}

	// lm head weight now
	lmHead := NewMatrix(dModel, vocabSize)
	lmHead.Randomize(scale)

	finalLnGamma := NewMatrix(1, dModel)
	finalLnBeta := NewMatrix(1, dModel)
	for j := 0; j < dModel; j++ {
		finalLnGamma.data[j] = 1.0
	}

	return &GPTModel{
		vocabSize:     vocabSize,
		dModel:        dModel,
		numLayers:     numLayers,
		numHeads:      numHeads,
		maxSeqLen:     maxSeqLen,
		tokenEmbed:    tokenEmbed,
		positionEmbed: positionEmbed,
		blocks:        blocks,
		lmHead:        lmHead,
		finalLnGamma:  finalLnGamma,
		finalLnBeta:   finalLnBeta,
		epsilon:       1e-5,
	}, nil
}

// foward executes the full forward pass of the GPT Model
// takes as input the tokenIDs
// outputs logits matrix of shape seqlen x vocabsize
func (m *GPTModel) Forward(tokenIDs []int) (*Matrix, error) {
	seqLen := len(tokenIDs)

	// get embeddings
	tokenEmbeddings, err := m.tokenEmbed.Forward(tokenIDs)
	if err != nil {
		return nil, err
	}

	// get pos embeddings
	posEmbeddings, err := m.positionEmbed.Forward(seqLen)
	if err != nil {
		return nil, err
	}

	// combine the 2
	hidden, err := tokenEmbeddings.MatrixAdd(posEmbeddings)
	if err != nil {
		return nil, err
	}

	// pass through the entire transformer block stack
	for i, block := range m.blocks {
		hidden, err = block.Forward(hidden)
		if err != nil {
			return nil, fmt.Errorf("error in transformer block %d: %v", i, err)
		}
	}

	// final layer normalization
	normed := hidden.LayerNorm(m.finalLnGamma, m.finalLnBeta, m.epsilon)

	// this is what projects to the vocabulary logits via LM HEAD
	logits, err := normed.MatrixMul(m.lmHead)
	if err != nil {
		return nil, err
	}

	return logits, nil
}

// forward method but with cache here
func (m *GPTModel) ForwardWithCache(tokenIDs []int, caches []*KvCache) (*Matrix, error) {
	seqLen := len(tokenIDs)

	tokenEmbeddings, err := m.tokenEmbed.Forward(tokenIDs)
	if err != nil {
		return nil, err
	}

	posOffset := 0
	if len(caches) > 0 && caches[0] != nil && caches[0].key != nil {
		posOffset = caches[0].key.rows
	}

	posEmbeddings, err := m.positionEmbed.ForwardAt(seqLen, posOffset)
	if err != nil {
		return nil, err
	}

	hidden, err := tokenEmbeddings.MatrixAdd(posEmbeddings)
	if err != nil {
		return nil, err
	}

	// making use of caching here
	for i, block := range m.blocks {
		hidden, err = block.ForwardWithCache(hidden, caches[i])
		if err != nil {
			return nil, fmt.Errorf("error in transformer block %d with cache: %v", i, err)
		}
	}

	normed := hidden.LayerNorm(m.finalLnGamma, m.finalLnBeta, m.epsilon)
	logits, err := normed.MatrixMul(m.lmHead)
	if err != nil {
		return nil, err
	}

	return logits, nil
}

type ModelActivations struct {
	inputTokenIDs []int
	tokenEmbeds   *Matrix
	blockActs     []*BlockActivations
	finalNormIn   *Matrix
	finalNormed   *Matrix
}

func (m *GPTModel) ForwardTrain(tokenIDs []int) (*Matrix, *ModelActivations, error) {
	seqLen := len(tokenIDs)

	tokenEmbeds, err := m.tokenEmbed.Forward(tokenIDs)
	if err != nil {
		return nil, nil, err
	}

	posEmbeds, err := m.positionEmbed.Forward(seqLen)
	if err != nil {
		return nil, nil, err
	}

	hidden, err := tokenEmbeds.MatrixAdd(posEmbeds)
	if err != nil {
		return nil, nil, err
	}

	blockActs := make([]*BlockActivations, m.numLayers)
	for i, block := range m.blocks {
		var acts *BlockActivations
		hidden, acts, err = block.ForwardSave(hidden)
		if err != nil {
			return nil, nil, fmt.Errorf("error in transformer block %d: %v", i, err)
		}
		blockActs[i] = acts
	}

	finalNormIn := hidden
	finalNormed := hidden.LayerNorm(m.finalLnGamma, m.finalLnBeta, m.epsilon)

	logits, err := finalNormed.MatrixMul(m.lmHead)
	if err != nil {
		return nil, nil, err
	}

	acts := &ModelActivations{
		inputTokenIDs: tokenIDs,
		tokenEmbeds:   tokenEmbeds,
		blockActs:     blockActs,
		finalNormIn:   finalNormIn,
		finalNormed:   finalNormed,
	}

	return logits, acts, nil
}

func (m *GPTModel) Backward(dLogits *Matrix, acts *ModelActivations) error {
	dFinalNormed, dLmHead, err := acts.finalNormed.MatrixMulBackward(m.lmHead, dLogits)
	if err != nil {
		return err
	}
	m.lmHead.AccumulateGrad(dLmHead)

	dHidden, dFinalLnGamma, dFinalLnBeta := LayerNormBackward(acts.finalNormIn, m.finalLnGamma, m.epsilon, dFinalNormed)
	m.finalLnGamma.AccumulateGrad(dFinalLnGamma)
	m.finalLnBeta.AccumulateGrad(dFinalLnBeta)

	for i := m.numLayers - 1; i >= 0; i-- {
		dHidden, err = m.blocks[i].Backward(acts.blockActs[i], dHidden)
		if err != nil {
			return fmt.Errorf("backward error in block %d: %v", i, err)
		}
	}

	seqLen := len(acts.inputTokenIDs)
	for i := 0; i < seqLen; i++ {
		tokenID := acts.inputTokenIDs[i]
		embStart := tokenID * m.dModel
		hidStart := i * m.dModel
		for d := 0; d < m.dModel; d++ {
			m.tokenEmbed.weights.grad[embStart+d] += dHidden.data[hidStart+d]
			m.positionEmbed.weights.grad[i*m.dModel+d] += dHidden.data[hidStart+d]
		}
	}

	return nil
}

// training related data

func (m *GPTModel) GetTrainableParameters() []*Matrix {
	var params []*Matrix

	// embeddings
	params = append(params, m.tokenEmbed.weights, m.positionEmbed.weights)
	for _, block := range m.blocks {
		// attention wegiths
		params = append(params, block.attention.wq, block.attention.wk, block.attention.wv, block.attention.wo)
		// ffn weights
		params = append(params, block.ffn.w1, block.ffn.w2)
		// layernorm param
		params = append(params, block.ln1Gamma, block.ln1Beta, block.ln2Gamma, block.ln2Beta)
	}
	params = append(params, m.finalLnGamma, m.finalLnBeta, m.lmHead)

	return params
}

func (m *GPTModel) ZeroGrads() {
	for _, p := range m.GetTrainableParameters() {
		if p != nil {
			p.ZeroGrad()
		}
	}
}

// train-step performs a single forward-backward-optimizer step on a token sequence
func (m *GPTModel) TrainStep(tokenIDs []int, optimizer *AdamW) (float32, error) {
	if len(tokenIDs) < 2 {
		return 0, fmt.Errorf("token sequence too short for training step")
	}

	// gradient reset
	m.ZeroGrads()

	inputIDs := tokenIDs[:len(tokenIDs)-1]
	targetIDs := tokenIDs[1:]
	seqLen := len(inputIDs)

	// forward pass
	logits, acts, err := m.ForwardTrain(inputIDs)
	if err != nil {
		return 0, fmt.Errorf("forward pass failed: %v", err)
	}

	// compute cross-entropy loss across positions
	var totalLoss float32
	// accumulate gradient matrices for the output of the model
	dLogits := NewMatrix(logits.rows, logits.cols)

	for i := 0; i < seqLen; i++ {
		// Extract logits row for token i
		rowStart := i * m.vocabSize
		rowEnd := rowStart + m.vocabSize
		rowLogitsMatrix, err := NewMatrixFromData(1, m.vocabSize, logits.data[rowStart:rowEnd])
		if err != nil {
			return 0, err
		}

		loss, gradRow, err := CrossEntropyLoss(rowLogitsMatrix, targetIDs[i])
		if err != nil {
			return 0, err
		}

		totalLoss += loss
		// copy gradient back into dLogits matrix
		copy(dLogits.data[rowStart:rowEnd], gradRow.data)
	}

	invSeqLen := float32(1.0) / float32(seqLen)
	for i := range dLogits.data {
		dLogits.data[i] *= invSeqLen
	}

	if err := m.Backward(dLogits, acts); err != nil {
		return 0, fmt.Errorf("backward pass failed: %v", err)
	}

	optimizer.timestep++

	// optimizer
	for _, param := range m.GetTrainableParameters() {
		if param != nil {
			optimizer.Step(param)
		}
	}

	return totalLoss / float32(seqLen), nil
}
