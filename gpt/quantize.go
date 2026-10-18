package gpt

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// int8 weight quantization
//
// the point is memory. a float32 weight costs 4 bytes, an int8 one costs 1,
// and the weights are almost the entire model -> gpt2-xl goes from about
// 6.5GB of matrices to under 2GB, which is the difference between running it
// and not.
//
// the scheme is "weight only, per output column". activations stay float32,
// which keeps this simple and keeps the error small. for y = x @ W, with W
// shaped (in, out), every output column j gets its own scale:
//
//	W[i][j] ~= scale[j] * q[i][j]      with q in [-127, 127]
//	y[r][j] = scale[j] * sum_i x[r][i] * q[i][j]
//
// the scale factors out of the sum, so the inner loop is still a plain
// multiply-accumulate and the scale is applied once per output element.
//
// per COLUMN rather than one scale for the whole tensor matters: a single
// scale has to cover the largest weight anywhere in the matrix, and gpt-2 has
// outliers, so most columns would end up using a tiny slice of the int8 range.
// per column gives each output feature its own dynamic range.
//
// the cost is speed. gonum's float32 matmul is hand written assembly and this
// is a Go loop, so expect it to be several times slower. that is the trade:
// four times less memory for a few times less throughput

// quantizedWeights is the int8 form of a weight matrix, stored on the Matrix
// itself so that nothing at the call sites has to change
type quantizedWeights struct {
	data []int8 // rows*cols, row major, same layout as the float32 version

	// one scale per column, so len(scales) == cols
	scales []float32
}

// IsQuantized reports whether this matrix holds int8 weights rather than
// float32 ones
func (m *Matrix) IsQuantized() bool {
	return m.quant != nil
}

// Quantize converts a weight matrix to int8 in place and releases the float32
// data.
//
// only do this to matrices that are exclusively used as the right hand side of
// a matmul -> the weights in attention and the mlp, plus the lm head. anything
// read element-wise still needs its float32 data:
//
//   - the token embeddings are a lookup table, not a matmul
//   - layernorm gammas and betas are read per element
//   - biases are added, not multiplied
//
// Quantize refuses to touch a matrix carrying gradients, since a quantized
// weight cannot be trained
func (m *Matrix) Quantize() error {
	if m.quant != nil {
		return nil
	}
	if m.data == nil {
		return fmt.Errorf("quantize: matrix has no float32 data to quantize")
	}
	if m.grad != nil {
		return fmt.Errorf("quantize: matrix has gradients, so it is being trained -> quantizing would strip the weights it needs")
	}
	if m.rows == 0 || m.cols == 0 {
		return fmt.Errorf("quantize: matrix is %dx%d", m.rows, m.cols)
	}

	scales := make([]float32, m.cols)
	data := make([]int8, m.rows*m.cols)

	// one pass for the per column maxima
	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		for j := 0; j < m.cols; j++ {
			v := float32(math.Abs(float64(m.data[rowStart+j])))
			if v > scales[j] {
				scales[j] = v
			}
		}
	}

	// turn the maxima into scales. an all-zero column keeps a scale of zero,
	// which reconstructs as zero and is exactly right
	for j := range scales {
		if scales[j] > 0 {
			scales[j] /= 127.0
		}
	}

	// and a second pass to actually quantize
	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		for j := 0; j < m.cols; j++ {
			scale := scales[j]
			if scale == 0 {
				continue
			}

			q := float64(m.data[rowStart+j]) / float64(scale)

			// round half away from zero, then clamp. the clamp should never
			// fire given the scale came from the column maximum, but a NaN or
			// an infinity in a corrupt checkpoint would otherwise wrap around
			rounded := math.Round(q)
			if rounded > 127 {
				rounded = 127
			} else if rounded < -127 {
				rounded = -127
			} else if math.IsNaN(rounded) {
				rounded = 0
			}

			data[rowStart+j] = int8(rounded)
		}
	}

	m.quant = &quantizedWeights{data: data, scales: scales}

	// release the float32 copy -> this is the entire point
	m.data = nil

	return nil
}

// Dequantize reconstructs the float32 approximation of a quantized matrix.
// used by the tests to measure how much precision the conversion costs, and
// it is what the maths in matMulQuantized is equivalent to
func (m *Matrix) Dequantize() (*Matrix, error) {
	if m.quant == nil {
		return nil, fmt.Errorf("dequantize: matrix is not quantized")
	}

	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		for j := 0; j < m.cols; j++ {
			result.data[rowStart+j] = float32(m.quant.data[rowStart+j]) * m.quant.scales[j]
		}
	}

	return result, nil
}

// matMulQuantized computes x @ w where w holds int8 weights.
//
// the loop order is r, k, j rather than the textbook r, j, k. that walks both
// the weight row and the output row sequentially, which matters a lot more
// than the instruction count -> the textbook order strides through the weight
// matrix by its row length and spends its life waiting on cache misses
func matMulQuantized(x *Matrix, w *Matrix) *Matrix {
	rows := x.rows
	inner := x.cols
	cols := w.cols

	result := NewMatrix(rows, cols)

	quant := w.quant
	scales := quant.scales

	// rows are independent, so hand them out to as many goroutines as there
	// are cores. this is the same trick the attention heads use
	workers := runtime.NumCPU()
	if workers > rows {
		workers = rows
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	chunk := (rows + workers - 1) / workers

	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}

		wg.Add(1)
		go func(from, to int) {
			defer wg.Done()

			for r := from; r < to; r++ {
				outRow := result.data[r*cols : (r+1)*cols]
				xRow := x.data[r*inner : (r+1)*inner]

				for k := 0; k < inner; k++ {
					xv := xRow[k]
					if xv == 0 {
						continue
					}
					qRow := quant.data[k*cols : (k+1)*cols]
					for j, qv := range qRow {
						outRow[j] += xv * float32(qv)
					}
				}

				// the per column scale comes out of the sum, so it is applied
				// once per output element rather than once per multiply
				for j := range outRow {
					outRow[j] *= scales[j]
				}
			}
		}(start, end)
	}

	wg.Wait()

	return result
}

// QuantizedBytesSaved reports how much the int8 weights are saving compared
// to holding them as float32, counting only what is actually quantized right
// now. zero for a plain float32 model
func (m *GPTModel) QuantizedBytesSaved() int {
	var saved int

	count := func(target *Matrix) {
		if target == nil || !target.IsQuantized() {
			return
		}
		// float32 would be 4 bytes an element. we pay 1 byte an element plus
		// one float32 scale per column
		saved += target.rows*target.cols*4 - (target.rows*target.cols + target.cols*4)
	}

	for _, block := range m.blocks {
		count(block.attention.wq)
		count(block.attention.wk)
		count(block.attention.wv)
		count(block.attention.wo)
		count(block.ffn.w1)
		count(block.ffn.w2)
	}
	count(m.lmHead)

	return saved
}

// QuantizeForInference converts every weight in the model that is only ever
// used as a matmul operand, and reports how many bytes it saved.
//
// NOTE: if you want the memory back, pass Quantize in GPT2Config instead so
// the conversion happens during the load. calling this afterwards leaves the
// steady state smaller but makes the PEAK worse, because the float32 weights
// and their int8 copies are both live for a moment
//
// deliberately leaves alone:
//   - the token and positional embeddings, which are row lookups
//   - layernorm gammas and betas, read per element
//   - every bias, which is added rather than multiplied
//
// those are a small share of the total anyway: for gpt2-medium the matrices
// this does convert are about 87% of the weight bytes
func (m *GPTModel) QuantizeForInference() (int, error) {
	var savedBytes int

	quantize := func(target *Matrix, what string) error {
		if target == nil || target.IsQuantized() {
			return nil
		}
		before := target.rows * target.cols * 4
		if err := target.Quantize(); err != nil {
			return fmt.Errorf("%s: %v", what, err)
		}
		// int8 data plus one float32 scale per column
		after := target.rows*target.cols + target.cols*4
		savedBytes += before - after
		return nil
	}

	for i, block := range m.blocks {
		for name, target := range map[string]*Matrix{
			"wq": block.attention.wq,
			"wk": block.attention.wk,
			"wv": block.attention.wv,
			"wo": block.attention.wo,
			"w1": block.ffn.w1,
			"w2": block.ffn.w2,
		} {
			if err := quantize(target, fmt.Sprintf("block %d %s", i, name)); err != nil {
				return savedBytes, err
			}
		}
	}

	if err := quantize(m.lmHead, "lm head"); err != nil {
		return savedBytes, err
	}

	return savedBytes, nil
}
