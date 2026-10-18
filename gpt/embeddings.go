package gpt

import "fmt"

// Embeddings implementation
// In the case of a Transformer
// We have 3 distinct steps
// 1. Token Embeddings (vocabulary size x hidden dim)
//
// Text tokens (like integers from BPE) are just indices. A token embedding table
// is esentially a giant lookup matrix where each row corresponds to a specific vocabulary
// token's desnse vector representation. So, we pass a sequence of token IDs and pull out
// their respective rows
//
// 2. Positional Embedding (max sequence length x hidden dim)
// Transformers process all tokens in parallel and are invariant to word order on their own, therefore
// in order to tell the model where a token is found in a sequencem we add a separate positional embedding
// vector to each token embedding vector
//
// 3. Combining Both
// Final input is going to be equal to Token Embedding + Positional Embedding, which will utilize MatrixAdd
//
// Inside a GPT Architecture, this is how it works:
// 1. pass the token ids(bpe) through the token embeddings layer to get a seq_len and a hidden_dim matrix
// 2. pass the sequence length through the Positional Embedding layer to get a matching seq_len and hidden dim matrix
// 3. call MatrixAdd from a resulting Token to combine them into the transformer input tensor
// ex:
//
// tokenEmbeds, err := tokenEmbeddingLayer.Forward(tokenIDs)
// posEmbeds, err := tokenEmbeddingLayer.GetPositionalEmbeddings(len(tokenIDs))
// inputTensor, err := tokenEmbeds.MatrixAdd(posEmbeds)

type Embedding struct {
	vocabSize int
	hiddenDim int
	weights   *Matrix
}

func NewEmbedding(vocabSize, hiddenDim int, scale float32) *Embedding {
	weights := NewMatrix(vocabSize, hiddenDim)
	weights.Randomize(scale)
	return &Embedding{
		vocabSize: vocabSize,
		hiddenDim: hiddenDim,
		weights:   weights,
	}
}

// this method performs the embedding lookup for a given sequence of tokenIDs
// takes as input a slice of token ids
// and returns as output the matrix of shape of len of token ids and hidden dim
func (e *Embedding) Forward(tokenIDs []int) (*Matrix, error) {
	seqLen := len(tokenIDs)
	result := NewMatrix(seqLen, e.hiddenDim)

	for i, id := range tokenIDs {
		if id < 0 || id >= e.vocabSize {
			return nil, fmt.Errorf("token ID: %d out of bounds for vocab size %d", id, e.vocabSize)
		}

		srcStart := id * e.hiddenDim
		srcEnd := srcStart + e.hiddenDim
		dstStart := i * e.hiddenDim

		copy(result.data[dstStart:dstStart+e.hiddenDim], e.weights.data[srcStart:srcEnd])
	}

	return result, nil
}

// this method will extract positional embeddings for a given sequence length
func (e *Embedding) GetPositionalEmbeddings(seqLen int) (*Matrix, error) {
	if seqLen > e.vocabSize {
		return nil, fmt.Errorf("sequence length %d exceeds max positional embedding size %d", seqLen, e.vocabSize)
	}

	// matrix slicing for the first seqLen positions
	result := NewMatrix(seqLen, e.hiddenDim)
	copy(result.data, e.weights.data[:seqLen*e.hiddenDim])

	return result, nil
}

type PositionalEmbedding struct {
	maxSeqLen int
	hiddenDim int
	weights   *Matrix
}

func NewPositionalEmbedding(maxSeqLen, hiddenDim int, scale float32) *PositionalEmbedding {
	weights := NewMatrix(maxSeqLen, hiddenDim)
	weights.Randomize(scale)
	return &PositionalEmbedding{
		maxSeqLen: maxSeqLen,
		hiddenDim: hiddenDim,
		weights:   weights,
	}
}

func (pe *PositionalEmbedding) Forward(seqLen int) (*Matrix, error) {
	return pe.ForwardAt(seqLen, 0)
}

func (pe *PositionalEmbedding) ForwardAt(seqLen, offset int) (*Matrix, error) {
	if offset+seqLen > pe.maxSeqLen {
		return nil, fmt.Errorf("sequence position %d exceeds max context length %d", offset+seqLen, pe.maxSeqLen)
	}
	result := NewMatrix(seqLen, pe.hiddenDim)
	srcStart := offset * pe.hiddenDim
	copy(result.data, pe.weights.data[srcStart:srcStart+seqLen*pe.hiddenDim])
	return result, nil
}
