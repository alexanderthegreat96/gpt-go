package gpt

// Matrix multiply is the single most expensive operation in both the forward and backward pass.
// Every attention projection (Q, K, V, O), every FFN layer, and the LM head are all matmuls.
// For a model with dModel=64 this is fine in pure Go, but as dModel grows the gap widens fast:
// a naive triple loop gets maybe 5-10% of peak CPU throughput, while a BLAS implementation
// (e.g. OpenBLAS via gonum) uses hand-written AVX/AVX-512 assembly, cache-blocking (tiling),
// and fused multiply-add (FMA) instructions to reach 80-90% of peak — often a 10-20x difference
// on large matrices.
//
// To keep the codebase dependency-free by default while still allowing that speedup, matrix
// multiply is abstracted behind the Backend interface. The pure-Go GoBackend is always compiled
// in and is the default. The BlasBackend lives in backend_blas.go behind a //go:build blas tag
// and is only compiled when you pass -tags blas. Swapping at runtime is one call: SetBackend.
// Everything else in the model — LayerNorm, GELU, Softmax, embeddings — is cheap enough that
// it stays in plain Go regardless of which backend is active.

type Backend interface {
	MatMul(a, b *Matrix) *Matrix
	MatMulTransB(a, b *Matrix) *Matrix
	MatMulTransA(a, b *Matrix) *Matrix
}

var activeBackend Backend = &GoBackend{}

func SetBackend(b Backend) {
	activeBackend = b
}

// default impl
// the idea was to have the system run purely in go
// but this would be slower than using blas backend
type GoBackend struct{}

func (g *GoBackend) MatMul(a, b *Matrix) *Matrix {
	c := NewMatrix(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		iA := i * a.cols
		iC := i * b.cols
		for k := 0; k < a.cols; k++ {
			aVal := a.data[iA+k]
			kB := k * b.cols
			for j := 0; j < b.cols; j++ {
				c.data[iC+j] += aVal * b.data[kB+j]
			}
		}
	}
	return c
}

func (g *GoBackend) MatMulTransB(a, b *Matrix) *Matrix {
	m, k, n := a.rows, a.cols, b.rows
	c := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		iA := i * k
		iC := i * n
		for j := 0; j < n; j++ {
			jB := j * k
			var sum float32
			for kk := 0; kk < k; kk++ {
				sum += a.data[iA+kk] * b.data[jB+kk]
			}
			c.data[iC+j] = sum
		}
	}
	return c
}

func (g *GoBackend) MatMulTransA(a, b *Matrix) *Matrix {
	k, m, n := a.rows, a.cols, b.cols
	c := NewMatrix(m, n)
	for kk := 0; kk < k; kk++ {
		kkA := kk * m
		kkB := kk * n
		for i := 0; i < m; i++ {
			aVal := a.data[kkA+i]
			iC := i * n
			for j := 0; j < n; j++ {
				c.data[iC+j] += aVal * b.data[kkB+j]
			}
		}
	}
	return c
}
