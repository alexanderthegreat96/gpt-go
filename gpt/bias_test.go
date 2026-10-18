package gpt

import (
	"math"
	"testing"
)

func TestAddRowBias(t *testing.T) {
	m, err := NewMatrixFromData(2, 3, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	bias, _ := NewMatrixFromData(1, 3, []float32{10, 20, 30})

	if err := m.AddRowBias(bias); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []float32{11, 22, 33, 14, 25, 36}
	for i, v := range expected {
		if m.data[i] != v {
			t.Errorf("at index %d: expected %f, got %f", i, v, m.data[i])
		}
	}
}

func TestAddRowBiasNilIsNoOp(t *testing.T) {
	m, _ := NewMatrixFromData(1, 2, []float32{1, 2})

	if err := m.AddRowBias(nil); err != nil {
		t.Fatalf("a nil bias should be a no-op, got %v", err)
	}
	if m.data[0] != 1 || m.data[1] != 2 {
		t.Errorf("a nil bias changed the data: %v", m.data)
	}
}

func TestAddRowBiasRejectsBadShape(t *testing.T) {
	m, _ := NewMatrixFromData(2, 3, []float32{1, 2, 3, 4, 5, 6})
	bias, _ := NewMatrixFromData(1, 2, []float32{1, 2})

	if err := m.AddRowBias(bias); err == nil {
		t.Errorf("expected an error when the bias width does not match")
	}
}

func TestColumnSums(t *testing.T) {
	m, _ := NewMatrixFromData(3, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
	})

	sums := m.ColumnSums()

	if sums.rows != 1 || sums.cols != 2 {
		t.Fatalf("expected 1x2, got %dx%d", sums.rows, sums.cols)
	}
	if sums.data[0] != 9 || sums.data[1] != 12 {
		t.Errorf("expected [9 12], got %v", sums.data)
	}
}

func TestGELUTanhMatchesTheErfVersionClosely(t *testing.T) {
	inputs := []float32{-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}
	m, _ := NewMatrixFromData(1, len(inputs), inputs)

	exact := m.GELU()
	approx := m.GELUTanh()

	var maxDiff float64
	for i := range inputs {
		d := math.Abs(float64(exact.data[i] - approx.data[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}

	if maxDiff > 0.005 {
		t.Errorf("the two GELUs should agree to within ~0.005, max diff was %g", maxDiff)
	}
	if maxDiff < 1e-6 {
		t.Errorf("the two GELUs came out identical (max diff %g), one of them is wrong", maxDiff)
	}

	// GELU(0) is 0 either way
	if math.Abs(float64(approx.data[4])) > 1e-6 {
		t.Errorf("expected GELUTanh(0) to be 0, got %f", approx.data[4])
	}
}

func TestGELUTanhBackwardMatchesNumericalGradient(t *testing.T) {
	// an analytic derivative is easy to get subtly wrong and a wrong one does
	// not crash, it just trains badly. so we check it against a finite
	// difference of the forward function
	inputs := []float32{-2, -1, -0.3, 0.3, 1, 2}
	m, _ := NewMatrixFromData(1, len(inputs), inputs)

	// an upstream gradient of all ones means the result is purely the local
	// derivative, which is what we want to compare
	ones := NewMatrix(1, len(inputs))
	for i := range ones.data {
		ones.data[i] = 1
	}

	analytic := GELUTanhBackward(m, ones)

	const h = 1e-4
	for i, x := range inputs {
		plus, _ := NewMatrixFromData(1, 1, []float32{x + h})
		minus, _ := NewMatrixFromData(1, 1, []float32{x - h})

		numerical := (plus.GELUTanh().data[0] - minus.GELUTanh().data[0]) / (2 * h)

		if math.Abs(float64(analytic.data[i]-numerical)) > 1e-3 {
			t.Errorf("at x=%g: analytic derivative %f, numerical %f", x, analytic.data[i], numerical)
		}
	}
}

func TestFeedForwardUsesTheConfiguredActivation(t *testing.T) {
	// the forward and backward passes have to agree on which GELU they are
	// using. if they ever diverge the gradients are quietly wrong, so both read
	// the one flag -> this just confirms the flag is actually honoured
	input, _ := NewMatrixFromData(1, 2, []float32{1, -1})

	ff := &FeedForward{
		dModel:      2,
		dInner:      2,
		w1:          identityMatrix(2),
		w2:          identityMatrix(2),
		useTanhGELU: false,
	}

	exactOut, err := ff.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ff.useTanhGELU = true
	tanhOut, err := ff.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if exactOut.data[0] == tanhOut.data[0] {
		t.Errorf("flipping useTanhGELU did not change the output, the flag is being ignored")
	}
}

// identityMatrix is a test helper so an FFN can be built that passes its input
// through untouched, leaving only the activation to observe
func identityMatrix(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.data[i*n+i] = 1
	}
	return m
}
