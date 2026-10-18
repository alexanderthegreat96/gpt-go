package gpt

import (
	"math"
	"testing"
)

func TestMatrixLayerNormDefault(t *testing.T) {
	m, _ := NewMatrixFromData(2, 3, []float32{1, 2, 3, 4, 5, 6})
	epsilon := float32(1e-5)

	res := m.LayerNorm(nil, nil, epsilon)

	if res.rows != 2 || res.cols != 3 {
		t.Fatalf("expected dimensions 2x3, got %dx%d", res.rows, res.cols)
	}

	row0 := res.GetRow(0)

	var sum float32
	for _, v := range row0 {
		sum += v
	}
	mean := sum / float32(len(row0))
	if math.Abs(float64(mean)) > 1e-4 {
		t.Errorf("expected normalized row mean to be ~0, got %f", mean)
	}

	var varSum float32
	for _, v := range row0 {
		diff := v - mean
		varSum += diff * diff
	}
	variance := varSum / float32(len(row0))
	if math.Abs(float64(variance-1.0)) > 1e-4 {
		t.Errorf("expected normalized row variance to be ~1, got %f", variance)
	}
}

func TestMatrixLayerNormWithGammaBeta(t *testing.T) {
	m, _ := NewMatrixFromData(1, 2, []float32{2, 4})

	gamma, _ := NewMatrixFromData(1, 2, []float32{2.0, 3.0})
	beta, _ := NewMatrixFromData(1, 2, []float32{1.0, 5.0})

	res := m.LayerNorm(gamma, beta, 1e-5)

	expected := []float32{-1.0, 8.0}

	for i, val := range expected {
		if math.Abs(float64(res.data[i]-val)) > 1e-3 {
			t.Errorf("at index %d: expected ~%f, got %f", i, val, res.data[i])
		}
	}
}
