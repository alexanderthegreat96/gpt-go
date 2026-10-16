package gpt

import (
	"fmt"
	"math/rand"
)

// This is the matrix implementation
// it contains basically, everything that we need for
// working with matrixes and paves the ground for the activations segment as well

// in order to handle back propagation, we need gradients
type Matrix struct {
	rows int
	cols int
	data []float32
	grad []float32
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float32, rows*cols),
		grad: make([]float32, rows*cols),
	}
}

func NewMatrixFromData(rows, cols int, data []float32) (*Matrix, error) {
	if len(data) != rows*cols {
		return nil, fmt.Errorf("data length %d does not match dimensions %dx%d (%d)", len(data), rows, cols, rows*cols)
	}
	return &Matrix{
		rows: rows,
		cols: cols,
		data: append([]float32(nil), data...),
		grad: make([]float32, rows*cols),
	}, nil
}

func (m *Matrix) ZeroGrad() {
	for i := range m.grad {
		m.grad[i] = 0.0
	}
}

func (m *Matrix) AccumulateGrad(dM *Matrix) {
	for i := range m.grad {
		m.grad[i] += dM.data[i]
	}
}

// fucking nightmare
// i've never done matrix mul in my life
// this is torture when you wanna do it in loops
// g fucking g
func (m *Matrix) MatrixMul(other *Matrix) (*Matrix, error) {
	if m.cols != other.rows {
		return nil, fmt.Errorf("dimension mismatch for MatMul: %dx%d and %dx%d", m.rows, m.cols, other.rows, other.cols)
	}
	return activeBackend.MatMul(m, other), nil
}

// much simpler than MatrixMul
// simply peforms element-wise addition of two matrices of same size
func (m *Matrix) MatrixAdd(other *Matrix) (*Matrix, error) {
	if m.rows != other.rows || m.cols != other.cols {
		return nil, fmt.Errorf("dimension mismatch for Add: %dx%d and %dx%d", m.rows, m.cols, other.rows, other.cols)
	}

	resultingMatrix := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		resultingMatrix.data[i] = m.data[i] + other.data[i]
	}

	return resultingMatrix, nil
}

// in self attention, we have to multiply the Query meatrix (Q)
// by the transposed Key matrix (kT). if K has dimensions N x D
// its transpose kT has dimensions D x N
// for an original matrix with Rows and Cols, and element at coodinate
// (i, j) moves (j, i) in the transposed matrix
func (m *Matrix) Transpose() *Matrix {
	resultingMatrix := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			// the old index in m is at: i * m.cols + j
			oldIdx := i*m.cols + j
			// the new index is at: j * result.cols + 1 -> result.cols is m.rows
			newIds := j*resultingMatrix.cols + i
			resultingMatrix.data[newIds] = m.data[oldIdx]
		}
	}

	return resultingMatrix
}

// in attention, we scale QKt dot product matrix by 1 over square root of dk
// where dK is the dimension of the key vectors. this is before passsing it to
// Softmax to keep the gradients stable
func (m *Matrix) Scale(scalar float32) {
	for i := range m.data {
		m.data[i] *= scalar
	}
}

// neural networks need random initial weights. since we cannot determine them
// from the get go, we simply generate them, so that the neurons don't learn
// identical features. i am utilizing math/rand package here to populate the weights
func (m *Matrix) Randomize(scale float32) {
	for i := range m.data {
		m.data[i] = (rand.Float32()*2 - 1) * scale
	}
}

// row-wise max and sum for softmax or layernorm
// basically, softmax and layernorm require computing statistics
// across a single row (ex: finding the max value in a row for numeric stability),
// or suming row values to get a probability distribution
func (m *Matrix) GetRow(rowId int) []float32 {
	start := rowId * m.cols
	end := start + m.cols
	rowCopy := make([]float32, m.cols)
	copy(rowCopy, m.data[start:end])
	return rowCopy
}

// row concatenation method which adds another matrix vertically -> row wise
func (m *Matrix) ConcatRows(other *Matrix) (*Matrix, error) {
	if m == nil {
		return other, nil
	}

	if other == nil {
		return m, nil
	}

	if m.cols != other.cols {
		return nil, fmt.Errorf("cannot concat matrices with different columns: %d vs %d", m.cols, other.cols)
	}

	newRows := m.rows + other.rows
	newData := make([]float32, len(m.data)+len(other.data))
	copy(newData, m.data)
	copy(newData[len(m.data):], other.data)

	return &Matrix{
		rows: newRows,
		cols: m.cols,
		data: newData,
	}, nil
}

func (m *Matrix) MatrixMulBackward(other *Matrix, dC *Matrix) (*Matrix, *Matrix, error) {
	if m.cols != other.rows {
		return nil, nil, fmt.Errorf("dimension mismatch for backward pass: %dx%d and %dx%d", m.rows, m.cols, other.rows, other.cols)
	}
	if dC.rows != m.rows || dC.cols != other.cols {
		return nil, nil, fmt.Errorf("dC dimensions %dx%d do not match output dimensions %dx%d", dC.rows, dC.cols, m.rows, m.cols)
	}

	// dA = dC * Bᵀ
	dA := activeBackend.MatMulTransB(dC, other)
	// dB = Aᵀ * dC
	dB := activeBackend.MatMulTransA(m, dC)

	return dA, dB, nil
}
