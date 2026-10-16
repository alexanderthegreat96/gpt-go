package gpt

import (
	"math"
)

func LayerNormBackward(input *Matrix, gamma *Matrix, epsilon float32, dOut *Matrix) (*Matrix, *Matrix, *Matrix) {
	N := input.cols
	dInput := NewMatrix(input.rows, input.cols)
	dGamma := NewMatrix(1, N)
	dBeta := NewMatrix(1, N)

	for i := 0; i < input.rows; i++ {
		rowStart := i * N
		row := input.data[rowStart : rowStart+N]
		dOutRow := dOut.data[rowStart : rowStart+N]

		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(N)

		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(N)
		std := float32(math.Sqrt(float64(variance + epsilon)))

		dxHat := make([]float32, N)
		for j := 0; j < N; j++ {
			g := float32(1.0)
			if gamma != nil {
				g = gamma.data[j]
			}
			dxHat[j] = dOutRow[j] * g
		}

		for j := 0; j < N; j++ {
			xHat := (row[j] - mean) / std
			dGamma.data[j] += dOutRow[j] * xHat
			dBeta.data[j] += dOutRow[j]
		}

		var dVar float32
		for j := 0; j < N; j++ {
			dVar += dxHat[j] * (row[j] - mean)
		}
		dVar *= -0.5 * float32(math.Pow(float64(variance+epsilon), -1.5))

		var dMean float32
		for j := 0; j < N; j++ {
			dMean += dxHat[j]
		}
		dMean = -dMean / std

		for j := 0; j < N; j++ {
			dInput.data[rowStart+j] = dxHat[j]/std + dVar*2*(row[j]-mean)/float32(N) + dMean/float32(N)
		}
	}

	return dInput, dGamma, dBeta
}

// unlike batch normalization, Layer Normalization normalizes across features
// of a single token row for each row doing:
// - computing the mean which is just averaging all the values across one row
// - computing the variance: measures how spread of the values are around the mean
// - normalizes: subtracts the mean and divides by the square root of the variants
// plus a small epsilon to prevent division errors
// - scales and shifts: multiply by a weight scale and add bias (as far as I read,
// in variance, gamma is set to 1.0 and bias beta is set to 0.0 initially or passed as params)
//
// small sidenote here, I am accepting gama and beta as matrices because this helps
// with testing and loading training weights later
func (m *Matrix) LayerNorm(gamma, beta *Matrix, epsilon float32) *Matrix {
	result := NewMatrix(m.rows, m.cols)

	for i := 0; i < m.rows; i++ {
		rowStart := i * m.cols
		rowEnd := rowStart + m.cols

		row := m.data[rowStart:rowEnd]

		// compute mean
		var sum float32 = 0.0
		for _, val := range row {
			sum += val
		}
		mean := sum / float32(m.cols)

		// compute variance
		var varSum float32 = 0.0
		for _, val := range row {
			diff := val - mean
			varSum += diff * diff
		}
		variance := varSum / float32(m.cols)

		// normalize and apply scale / shift
		stdDev := float32(math.Sqrt(float64(variance + epsilon)))
		for j := 0; j < m.cols; j++ {
			g := float32(1.0)
			b := float32(0.0)
			if gamma != nil {
				g = gamma.data[j]
			}

			if beta != nil {
				b = beta.data[j]
			}

			normalized := (row[j] - mean) / stdDev
			result.data[rowStart+j] = (normalized * g) + b
		}
	}

	return result
}
