package gpt

type Backend interface {
	GetBackendName() string
	MatMul(a, b *Matrix) *Matrix
	MatMulTransB(a, b *Matrix) *Matrix
	MatMulTransA(a, b *Matrix) *Matrix
}

var activeBackend Backend = &GoBackend{}

// small helper to aid with unit testing
// in the context of MLs, build tags are preffered
// over setters
func SetBackend(b Backend) {
	activeBackend = b
}

// default impl
// the idea was to have the system run purely in go
// but this would be slower than using blas backend
type GoBackend struct{}

func (g *GoBackend) GetBackendName() string {
	return "Pure GO (Native)"
}

// regular go backend
// i implemented these by hand, and, although
// they work, they are extremely slow
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
