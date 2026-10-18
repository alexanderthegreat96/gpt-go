package gpt

import (
	"math"
	"testing"
)

func TestAdamWOptimizer(t *testing.T) {
	param, err := NewMatrixFromData(1, 1, []float32{1.0})
	if err != nil {
		t.Fatalf("failed to create param matrix: %v", err)
	}

	// gradients are lazily allocated now, so poking param.grad directly would
	// index a nil slice. going through AccumulateGrad is both correct and
	// closer to what a real training step does
	incoming, err := NewMatrixFromData(1, 1, []float32{0.5})
	if err != nil {
		t.Fatalf("failed to create gradient matrix: %v", err)
	}
	param.AccumulateGrad(incoming)

	optimizer := NewAdamW(0.01, 0.9, 0.999, 1e-8, 0.01)

	initialVal := param.data[0]
	optimizer.Step(param)
	updatedVal := param.data[0]

	if math.Abs(float64(initialVal-updatedVal)) < 1e-6 {
		t.Errorf("expected parameter to update after AdamW step, but value remained %f", updatedVal)
	}
}
