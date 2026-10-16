package gpt

import (
	"fmt"
	"math"
)

// cross entropy implementation
// this computes the loss and returns the gradient of logits
// for a simple token prediction
// logits: raw model output of shape [1, vocabSize]
// targetId: the actual tokenId

func CrossEntropyLoss(logits *Matrix, targetID int) (float32, *Matrix, error) {
	if targetID < 0 || targetID >= logits.cols {
		return 0, nil, fmt.Errorf("target token ID: %d out of bounds for vocab size %d", targetID, logits.cols)
	}

	// compute softmax probs
	probs := logits.Softmax()

	// compute loss
	targetProb := probs.data[targetID]
	if targetProb < 1e-15 {
		targetProb = 1e-15
	}

	loss := float32(-math.Log(float64(targetProb)))

	// gradient of softmax
	gradData := make([]float32, len(probs.data))
	copy(gradData, probs.data)
	gradData[targetID] -= 1.0

	gradMatrix, err := NewMatrixFromData(logits.rows, logits.cols, gradData)
	if err != nil {
		return 0, nil, err
	}

	return loss, gradMatrix, nil
}
