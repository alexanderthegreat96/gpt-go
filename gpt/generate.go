package gpt

// Inference and Text Generation Loop
// this is where it all comes together
// in order of execution
// 1. extract the last token logits which happens as a result of passing
// a sequence of length N into model.Forward(tokenIDs), we then get a (seqLen x vocabSize) matrix.
// Now, in order to predict the next token, we only care about the logits at the very end, which is
// seqLen-1
// 2. the next step is sampling and argmax. these raw logits are passed through softmax to turn them
// into probabilities, pick up the highest probable token and return it
// 3. the generation loop is the final stage where we append the predicated token ID into the input
// slice, feed it back into the model and repeat for N steps to generate a full sequence

import (
	"math/rand"
	"sort"
)

type GenerateConfig struct {
	Temperature       float32
	TopK              int
	RepetitionPenalty float32
}

func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		Temperature:       1.0,
		TopK:              40,
		RepetitionPenalty: 1.0,
	}
}

func (m *GPTModel) Generate(promptTokenIDs []int, maxNewTokens int, cfg GenerateConfig) ([]int, error) {
	currentTokens := make([]int, len(promptTokenIDs))
	copy(currentTokens, promptTokenIDs)

	// i am initializing an empty cache buffer here
	caches := make([]*KvCache, m.numLayers)
	for i := range caches {
		caches[i] = &KvCache{}
	}

	logits, err := m.ForwardWithCache(currentTokens, caches)
	if err != nil {
		return nil, err
	}

	for step := 0; step < maxNewTokens; step++ {
		seqLen := logits.rows
		lastRowStart := (seqLen - 1) * m.vocabSize

		rowData := make([]float32, m.vocabSize)
		copy(rowData, logits.data[lastRowStart:lastRowStart+m.vocabSize])

		if cfg.RepetitionPenalty != 1.0 {
			seen := make(map[int]bool)
			for _, id := range currentTokens {
				if !seen[id] {
					seen[id] = true
					if rowData[id] > 0 {
						rowData[id] /= cfg.RepetitionPenalty
					} else {
						rowData[id] *= cfg.RepetitionPenalty
					}
				}
			}
		}

		var nextToken int

		if cfg.Temperature <= 0 {
			var maxVal float32 = -1e38
			for i, v := range rowData {
				if v > maxVal {
					maxVal = v
					nextToken = i
				}
			}
		} else {
			if cfg.Temperature != 1.0 {
				for i := range rowData {
					rowData[i] /= cfg.Temperature
				}
			}

			if cfg.TopK > 0 && cfg.TopK < m.vocabSize {
				sorted := make([]float32, len(rowData))
				copy(sorted, rowData)
				sort.Slice(sorted, func(a, b int) bool { return sorted[a] > sorted[b] })
				threshold := sorted[cfg.TopK-1]
				for i, v := range rowData {
					if v < threshold {
						rowData[i] = -1e9
					}
				}
			}

			logMat, err := NewMatrixFromData(1, m.vocabSize, rowData)
			if err != nil {
				return nil, err
			}
			probs := logMat.Softmax()

			r := rand.Float32()
			var cumulative float32
			nextToken = m.vocabSize - 1
			for i, p := range probs.data {
				cumulative += p
				if r < cumulative {
					nextToken = i
					break
				}
			}
		}

		currentTokens = append(currentTokens, nextToken)

		// token caching
		if caches[0].key != nil && caches[0].key.rows >= m.maxSeqLen {
			break
		}

		logits, err = m.ForwardWithCache([]int{nextToken}, caches)
		if err != nil {
			return nil, err
		}
	}

	return currentTokens, nil
}
