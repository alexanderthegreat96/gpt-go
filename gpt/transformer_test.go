package gpt

import "testing"

func TestFeedForwardForward(t *testing.T) {
	dModel := 8
	scale := float32(0.1)

	ffn := NewFeedForward(dModel, scale)

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

	output, err := ffn.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error during FFN forward: %v", err)
	}

	if output.rows != seqLen || output.cols != dModel {
		t.Errorf("expected output dimensions %dx%d, got %dx%d", seqLen, dModel, output.rows, output.cols)
	}
}

func TestTransformerBlockForward(t *testing.T) {
	dModel := 8
	numHeads := 2
	scale := float32(0.1)

	block, err := NewTransformerBlock(dModel, numHeads, scale)
	if err != nil {
		t.Fatalf("failed to create transformer block: %v", err)
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

	output, err := block.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error during TransformerBlock forward: %v", err)
	}

	if output.rows != seqLen || output.cols != dModel {
		t.Errorf("expected output dimensions %dx%d, got %dx%d", seqLen, dModel, output.rows, output.cols)
	}
}
