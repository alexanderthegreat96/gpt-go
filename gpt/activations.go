package gpt

import (
	"math"
)

// activations segment
// where GELU and Softmax Live

// (i am not much of a statistics guy myself)
// but what GELU is doing, it simply applies
// Gaussian Error Linera Unit Activation function element-wise
// the formula looks like this: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// NOT using ReLU simply because in GPT architectures, GELU provides a smoother
// non-monotonic curve that allows small negative values to pass through with a low probability
// rather than zeroing them out entirely. Good for gradient flow during training
func GELUBackward(input *Matrix, dOut *Matrix) *Matrix {
	result := NewMatrix(input.rows, input.cols)
	for i, val := range input.data {
		x := float64(val)
		prime := 0.5*(1.0+math.Erf(x/math.Sqrt2)) + x*math.Exp(-x*x/2.0)/math.Sqrt(2.0*math.Pi)
		result.data[i] = dOut.data[i] * float32(prime)
	}
	return result
}

func SoftmaxBackward(probs *Matrix, dOut *Matrix) *Matrix {
	result := NewMatrix(probs.rows, probs.cols)
	for i := 0; i < probs.rows; i++ {
		rowStart := i * probs.cols
		var dot float32
		for j := 0; j < probs.cols; j++ {
			dot += dOut.data[rowStart+j] * probs.data[rowStart+j]
		}
		for j := 0; j < probs.cols; j++ {
			result.data[rowStart+j] = probs.data[rowStart+j] * (dOut.data[rowStart+j] - dot)
		}
	}
	return result
}

func (m *Matrix) GELU() *Matrix {
	result := NewMatrix(m.rows, m.cols)
	for i, val := range m.data {
		x := float64(val)
		geluVal := 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
		result.data[i] = float32(geluVal)
	}

	return result
}

// softmax utilizes a row by row approach to convert raw logits to probabilities
// also, in order to prevent floating-point overflow when exponential values get
// too large, we subtract the maximum value of each row before applying the exponential
func (m *Matrix) Softmax() *Matrix {
	result := NewMatrix(m.rows, m.cols)

	for i := 0; i < m.rows; i++ {
		// extract the current row
		rowStart := i * m.cols
		rowEnd := rowStart + m.cols
		row := m.data[rowStart:rowEnd]

		// find max value in the row -> numeric stability
		maxVal := row[0]
		for _, val := range row {
			if val > maxVal {
				maxVal = val
			}
		}

		// compute exponentials and sum them up
		sumExp := float32(0.0)
		exps := make([]float32, m.cols)
		for j, val := range row {
			expVal := float32(math.Exp(float64(val - maxVal)))
			exps[j] = expVal
			sumExp += expVal
		}

		// normalize by dividing each exponential by the sum of exponentials
		for j := 0; j < m.cols; j++ {
			result.data[rowStart+j] = exps[j] / sumExp
		}
	}

	return result
}
