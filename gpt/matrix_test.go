package gpt

import "testing"

func TestMatrixMultiplication(t *testing.T) {
	a := NewMatrix(2, 2)
	a.data = []float32{1, 2, 3, 4}

	b := NewMatrix(2, 2)
	b.data = []float32{5, 6, 7, 8}

	c, err := a.MatrixMul(b)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []float32{19, 22, 43, 50}
	if len(c.data) != len(expected) {
		t.Fatalf("expected data length %d, got %d", len(expected), len(c.data))
	}

	for i, val := range expected {
		if c.data[i] != val {
			t.Errorf("at index %d: expected %f, got %f", i, val, c.data[i])
		}
	}
}

func TestMatrixAdd(t *testing.T) {
	a := NewMatrix(2, 2)
	a.data = []float32{1, 2, 3, 4}

	b := NewMatrix(2, 2)
	b.data = []float32{10, 20, 30, 40}

	c, err := a.MatrixAdd(b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []float32{11, 22, 33, 44}
	if len(c.data) != len(expected) {
		t.Fatalf("expected data length %d, got %d", len(expected), len(c.data))
	}

	for i, val := range expected {
		if c.data[i] != val {
			t.Errorf("at index %d: expected %f, got %f", i, val, c.data[i])
		}
	}
}

func TestMatrixAddDimensionMismatch(t *testing.T) {
	a := NewMatrix(2, 2)
	b := NewMatrix(3, 3)

	_, err := a.MatrixAdd(b)
	if err == nil {
		t.Error("expected error due to dimension mismatch, got nil")
	}
}

func TestMatrixTranspose(t *testing.T) {
	m, err := NewMatrixFromData(2, 3, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	transposed := m.Transpose()

	if transposed.rows != 3 || transposed.cols != 2 {
		t.Fatalf("expected dimensions 3x2, got %dx%d", transposed.rows, transposed.cols)
	}

	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, val := range expected {
		if transposed.data[i] != val {
			t.Errorf("at index %d: expected %f, got %f", i, val, transposed.data[i])
		}
	}
}

func TestMatrixScale(t *testing.T) {
	m, _ := NewMatrixFromData(2, 2, []float32{2, 4, 6, 8})
	m.Scale(0.5)

	expected := []float32{1, 2, 3, 4}
	for i, val := range expected {
		if m.data[i] != val {
			t.Errorf("at index %d: expected %f, got %f", i, val, m.data[i])
		}
	}
}

func TestMatrixRandomize(t *testing.T) {
	m := NewMatrix(3, 3)
	scale := float32(0.5)
	m.Randomize(scale)

	for i, val := range m.data {
		if val < -scale || val > scale {
			t.Errorf("at index %d: value %f out of bounds [-%f, %f]", i, val, scale, scale)
		}
	}
}

func TestMatrixGetRow(t *testing.T) {
	m, _ := NewMatrixFromData(2, 3, []float32{1, 2, 3, 4, 5, 6})

	row1 := m.GetRow(1)
	expected := []float32{4, 5, 6}

	if len(row1) != len(expected) {
		t.Fatalf("expected row length %d, got %d", len(expected), len(row1))
	}

	for i, val := range expected {
		if row1[i] != val {
			t.Errorf("at index %d: expected %f, got %f", i, val, row1[i])
		}
	}
}

func TestMatrixMulBackward(t *testing.T) {
	A, _ := NewMatrixFromData(2, 2, []float32{1.0, 2.0, 3.0, 4.0})
	B, _ := NewMatrixFromData(2, 2, []float32{5.0, 6.0, 7.0, 8.0})
	dC, _ := NewMatrixFromData(2, 2, []float32{1.0, 1.0, 1.0, 1.0})

	dA, dB, err := A.MatrixMulBackward(B, dC)
	if err != nil {
		t.Fatalf("unexpected error in MatrixMulBackward: %v", err)
	}

	if dA.rows != 2 || dA.cols != 2 {
		t.Errorf("dA dimensions incorrect: %dx%d", dA.rows, dA.cols)
	}
	if dB.rows != 2 || dB.cols != 2 {
		t.Errorf("dB dimensions incorrect: %dx%d", dB.rows, dB.cols)
	}
}
