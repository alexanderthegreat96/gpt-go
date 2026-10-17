//go:build blas

package gpt

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// Making use of the BLAS Backend
// for matrix multiply to gonum/blas32
// Build requirement: go run/build -tags blas ...
type BlasBackend struct{}

func GetActiveBackendName() string {
	return "Gonum BLAS"
}

func init() {
	SetBackend(&BlasBackend{})
}

func general(m *Matrix) blas32.General {
	stride := m.cols
	if stride < 1 {
		stride = 1
	}
	return blas32.General{
		Rows:   m.rows,
		Cols:   m.cols,
		Stride: stride,
		Data:   m.data,
	}
}

func (bl *BlasBackend) MatMul(a, b *Matrix) *Matrix {
	c := NewMatrix(a.rows, b.cols)
	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, general(a), general(b), 0, general(c))
	return c
}

func (bl *BlasBackend) MatMulTransB(a, b *Matrix) *Matrix {
	c := NewMatrix(a.rows, b.rows)
	blas32.Gemm(blas.NoTrans, blas.Trans, 1, general(a), general(b), 0, general(c))
	return c
}

func (bl *BlasBackend) MatMulTransA(a, b *Matrix) *Matrix {
	c := NewMatrix(a.cols, b.cols)
	blas32.Gemm(blas.Trans, blas.NoTrans, 1, general(a), general(b), 0, general(c))
	return c
}
