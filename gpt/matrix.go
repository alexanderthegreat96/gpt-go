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

	// when non-nil this matrix holds int8 weights instead of float32 ones,
	// and data is released. keeping it here rather than in a separate type
	// means MatrixMul is the ONLY place that has to know -> every existing
	// input.MatrixMul(weight) call site works unchanged.
	//
	// only ever set on weights used purely as a matmul operand. see
	// quantize.go for which those are and why
	quant *quantizedWeights
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float32, rows*cols),
		// grad stays nil until something back propagates it
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
		// grad stays nil until something back propagates it
	}, nil
}

// a matrix that never had a gradient has nothing to zero, and ranging over a
// nil slice is a no-op, so this needs no nil check of its own
func (m *Matrix) ZeroGrad() {
	for i := range m.grad {
		m.grad[i] = 0.0
	}
}

func (m *Matrix) AccumulateGrad(dM *Matrix) {
	// this call is not optional. without it, a matrix whose grad is still nil
	// ranges over an empty slice and this whole function silently does
	// NOTHING -> every weight gradient gets thrown away, training runs, the
	// loss looks vaguely plausible and the model learns nothing at all.
	// no panic, no error. ask me how I know
	m.ensureGrad()

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

	// the one and only place that knows about quantization. the weights get
	// converted to int8, activations stay float32, and this picks the matching
	// kernel -> nothing in attention or the mlp changes
	if other.IsQuantized() {
		if m.IsQuantized() {
			return nil, fmt.Errorf("MatMul: both operands are quantized, only the weights should be")
		}
		return matMulQuantized(m, other), nil
	}
	if m.IsQuantized() {
		return nil, fmt.Errorf("MatMul: the left operand is quantized, that should only ever be the weights on the right")
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

// AddRowBias adds a 1 x cols bias vector to every row of the matrix, in place.
//
// we never needed this before because nothing we build from scratch uses bias
// terms. gpt-2 does, on every single linear layer -> it was written with
// Conv1D rather than nn.Linear and the forward pass is literally
// x @ weight + bias. that means four bias vectors per block (c_attn, the
// attention c_proj, mlp c_fc, mlp c_proj) and without them the output is not
// approximately right, it is just wrong
//
// a nil bias is a deliberate no-op. that way every existing call site keeps
// working untouched and only the pretrained path pays for this
func (m *Matrix) AddRowBias(bias *Matrix) error {
	if bias == nil {
		return nil
	}

	if bias.rows != 1 || bias.cols != m.cols {
		return fmt.Errorf("bias shape %dx%d cannot broadcast over %dx%d", bias.rows, bias.cols, m.rows, m.cols)
	}

	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		for j := 0; j < m.cols; j++ {
			m.data[rowStart+j] += bias.data[j]
		}
	}

	return nil
}

// ColumnSums collapses a matrix down to a single 1 x cols row by summing each
// column. this is exactly the gradient of a row-broadcast bias: the same bias
// vector got added to every row on the way forward, so on the way back every
// row contributes to it
func (m *Matrix) ColumnSums() *Matrix {
	result := NewMatrix(1, m.cols)
	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		for j := 0; j < m.cols; j++ {
			result.data[j] += m.data[rowStart+j]
		}
	}
	return result
}

// accumulateBiasGrad folds the column sums of an output gradient into a bias
// term, skipping the whole thing when that bias does not exist. keeps the
// backward passes from being littered with nil checks
func accumulateBiasGrad(bias *Matrix, dOut *Matrix) {
	if bias == nil {
		return
	}
	bias.AccumulateGrad(dOut.ColumnSums())
}

// lazyly allocate the gradient buffer on first write
// before making use of this, every matrix used to allocate a gradient slice
// the same size as its data, wether anything or not would backpropagade it
// now, if you train something small, it's fine.
// but for inference, that is simply wasted space
func (m *Matrix) ensureGrad() {
	if m.grad == nil {
		m.grad = make([]float32, len(m.data))
	}
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
	// a quantized weight has no float32 data left to differentiate, and the
	// int8 values are not what the gradient is with respect to. fail loudly
	// rather than nil dereference somewhere deeper
	if m.IsQuantized() || other.IsQuantized() {
		return nil, nil, fmt.Errorf("MatMulBackward: operand is quantized -> a quantized model is for inference only")
	}

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

// the matrix multiplication and addition methods are bound within backend.go and backend_blas.go
// the reason is quite simple: blas is faster due to custom assembly
