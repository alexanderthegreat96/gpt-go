package gpt

import (
	"math"
	"testing"
)

func TestMatrixGELU(t *testing.T) {
	m, _ := NewMatrixFromData(1, 3, []float32{0.0, -1.0, 1.0})
	result := m.GELU()

	if len(result.data) != 3 {
		t.Fatalf("expected length 3, got %d", len(result.data))
	}

	if math.Abs(float64(result.data[0])) > 1e-5 {
		t.Errorf("expected GELU(0) to be ~0, got %f", result.data[0])
	}

	expectedNeg := float32(0.5 * -1.0 * (1.0 + math.Erf(-1.0/math.Sqrt2)))
	if math.Abs(float64(result.data[1]-expectedNeg)) > 1e-5 {
		t.Errorf("expected GELU(-1) to be ~%f, got %f", expectedNeg, result.data[1])
	}

	expectedPos := float32(0.5 * 1.0 * (1.0 + math.Erf(1.0/math.Sqrt2)))
	if math.Abs(float64(result.data[2]-expectedPos)) > 1e-5 {
		t.Errorf("expected GELU(1) to be ~%f, got %f", expectedPos, result.data[2])
	}
}

func TestMatrixSoftmax(t *testing.T) {
	m, _ := NewMatrixFromData(2, 2, []float32{
		0.0, 0.0,
		1.0, 2.0,
	})

	result := m.Softmax()

	row1Sum := result.data[0] + result.data[1]
	if math.Abs(float64(row1Sum-1.0)) > 1e-5 {
		t.Errorf("row 1 probabilities do not sum to 1.0: got %f", row1Sum)
	}
	if math.Abs(float64(result.data[0]-0.5)) > 1e-5 || math.Abs(float64(result.data[1]-0.5)) > 1e-5 {
		t.Errorf("expected [0.5, 0.5] for row 1, got [%f, %f]", result.data[0], result.data[1])
	}

	row2Sum := result.data[2] + result.data[3]
	if math.Abs(float64(row2Sum-1.0)) > 1e-5 {
		t.Errorf("row 2 probabilities do not sum to 1.0: got %f", row2Sum)
	}
	if result.data[2] >= result.data[3] {
		t.Errorf("expected second element to be larger in row 2 softmax")
	}
}
