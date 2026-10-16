package gpt

import "testing"

func TestCrossEntropyLoss(t *testing.T) {
	logits, err := NewMatrixFromData(1, 3, []float32{2.0, 1.0, 0.1})
	if err != nil {
		t.Fatalf("failed to create logits matrix: %v", err)
	}

	targetID := 0

	loss, grad, err := CrossEntropyLoss(logits, targetID)
	if err != nil {
		t.Fatalf("unexpected error in CrossEntropyLoss: %v", err)
	}

	if loss <= 0 {
		t.Errorf("expected positive loss, got %f", loss)
	}

	if grad.rows != logits.rows || grad.cols != logits.cols {
		t.Errorf("gradient dimensions mismatch: got %dx%d, expected %dx%d", grad.rows, grad.cols, logits.rows, logits.cols)
	}

	if grad.data[targetID] >= 0 {
		t.Errorf("expected negative gradient at target index %d, got %f", targetID, grad.data[targetID])
	}
}
