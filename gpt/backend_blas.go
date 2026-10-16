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

func (bl *BlasBackend) MatMul(a, b *Matrix) *Matrix {
	m, k, n := a.rows, a.cols, b.cols
	c := NewMatrix(m, n)
	blas32.Gemm(blas.NoTrans, blas.NoTrans, m, n, k,
		1.0, a.data, k,
		b.data, n,
		0.0, c.data, n)
	return c
}

func (bl *BlasBackend) MatMulTransB(a, b *Matrix) *Matrix {
	m, k, n := a.rows, a.cols, b.rows
	c := NewMatrix(m, n)
	blas32.Gemm(blas.NoTrans, blas.Trans, m, n, k,
		1.0, a.data, k,
		b.data, k,
		0.0, c.data, n)
	return c
}

func (bl *BlasBackend) MatMulTransA(a, b *Matrix) *Matrix {
	k, m, n := a.rows, a.cols, b.cols
	c := NewMatrix(m, n)
	blas32.Gemm(blas.Trans, blas.NoTrans, m, n, k,
		1.0, a.data, m,
		b.data, n,
		0.0, c.data, n)
	return c
}
