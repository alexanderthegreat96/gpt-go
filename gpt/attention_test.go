package gpt

import "testing"

func TestNewCausalAttention(t *testing.T) {
	ca, err := NewCausalAttention(16, 4, 0.1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ca.dModel != 16 || ca.numHeads != 4 || ca.headDim != 4 {
		t.Errorf("unexpected parameters: dModel=%d, numHeads=%d, headDim=%d", ca.dModel, ca.numHeads, ca.headDim)
	}

	_, err = NewCausalAttention(15, 4, 0.1)
	if err == nil {
		t.Error("expected error when dModel is not divisible by numHeads, got nil")
	}
}

func TestCausalAttentionForward(t *testing.T) {
	dModel := 8
	numHeads := 2
	scale := float32(0.1)

	ca, err := NewCausalAttention(dModel, numHeads, scale)
	if err != nil {
		t.Fatalf("failed to create attention layer: %v", err)
	}

	seqLen := 3
	inputData := []float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
	}
	input, err := NewMatrixFromData(seqLen, dModel, inputData)
	if err != nil {
		t.Fatalf("failed to create input matrix: %v", err)
	}

	output, err := ca.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error during attention forward pass: %v", err)
	}

	if output.rows != seqLen || output.cols != dModel {
		t.Errorf("expected output dimensions %dx%d, got %dx%d", seqLen, dModel, output.rows, output.cols)
	}

	expectedDataLen := seqLen * dModel
	if len(output.data) != expectedDataLen {
		t.Errorf("expected data length %d, got %d", expectedDataLen, len(output.data))
	}
}
