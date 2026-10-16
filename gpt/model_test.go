package gpt

import "testing"

func TestGPTModelForward(t *testing.T) {
	vocabSize := 100
	dModel := 16
	numLayers := 2
	numHeads := 4
	scale := float32(0.1)

	model, err := NewGPTModel(vocabSize, dModel, numLayers, numHeads, 512, scale)
	if err != nil {
		t.Fatalf("failed to create GPT model: %v", err)
	}

	tokenIDs := []int{5, 12, 42, 88}
	seqLen := len(tokenIDs)

	logits, err := model.Forward(tokenIDs)
	if err != nil {
		t.Fatalf("unexpected error during GPT model forward pass: %v", err)
	}

	if logits.rows != seqLen || logits.cols != vocabSize {
		t.Errorf("expected logits dimensions %dx%d, got %dx%d", seqLen, vocabSize, logits.rows, logits.cols)
	}

	expectedDataLen := seqLen * vocabSize
	if len(logits.data) != expectedDataLen {
		t.Errorf("expected logits data length %d, got %d", expectedDataLen, len(logits.data))
	}
}
