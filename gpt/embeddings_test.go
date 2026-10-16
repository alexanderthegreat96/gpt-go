package gpt

import "testing"

func TestEmbeddingForward(t *testing.T) {
	vocabSize := 5
	hiddenDim := 4
	scale := float32(0.1)

	emb := NewEmbedding(vocabSize, hiddenDim, scale)

	tokenIDs := []int{2, 0, 4}
	res, err := emb.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error during embedding forward: %v", err)
	}

	if res.rows != len(tokenIDs) || res.cols != hiddenDim {
		t.Fatalf("expected dimensions 3x4, got %dx%d", res.rows, res.cols)
	}

	weightRow2Start := 2 * hiddenDim
	for j := 0; j < hiddenDim; j++ {
		if res.data[j] != emb.weights.data[weightRow2Start+j] {
			t.Errorf("at col %d: expected embedding value %f, got %f", j, emb.weights.data[weightRow2Start+j], res.data[j])
		}
	}
}

func TestEmbeddingOutOfBounds(t *testing.T) {
	emb := NewEmbedding(5, 4, 0.1)

	_, err := emb.Forward([]int{1, 10})
	if err == nil {
		t.Error("expected error for out-of-bounds token ID, got nil")
	}
}

func TestGetPositionalEmbeddings(t *testing.T) {
	maxSeqLen := 10
	hiddenDim := 4
	emb := NewEmbedding(maxSeqLen, hiddenDim, 0.1)

	seqLen := 3
	posEmbeds, err := emb.GetPositionalEmbeddings(seqLen)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if posEmbeds.rows != seqLen || posEmbeds.cols != hiddenDim {
		t.Fatalf("expected dimensions 3x4, got %dx%d", posEmbeds.rows, posEmbeds.cols)
	}
}
