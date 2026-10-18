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

	// StopTokenIDs ends generation early if any of these comes up.
	//
	// gpt-2 needs this: it emits 50256 to mean "this document is finished",
	// and without stopping you get a fresh unrelated document stapled onto
	// your answer. a slice rather than a single id on purpose -> nil is the
	// zero value and means "do not stop", so every existing caller keeps the
	// behaviour it had. an int field would have made id 0 a stop token by
	// accident for the models we train ourselves
	StopTokenIDs []int

	// SlideContext keeps generating once the context window is full, instead
	// of stopping dead at maxSeqLen.
	//
	// it has to rebuild the KV cache rather than just dropping old rows,
	// because gpt-2's positional embeddings are ABSOLUTE and baked into the
	// cached keys at the input. drop the first row and the remaining entries
	// still claim positions 1..n while the next token would be handed position
	// n again -> two tokens at the same position and a model that quietly
	// loses track of word order.
	//
	// so when the window fills we keep the most recent half, clear the caches
	// and re-run a forward pass over the retained tokens, which numbers their
	// positions from zero again. that costs one prefill every maxSeqLen/2
	// tokens. keeping all but one token instead would be correct too, and
	// would re-prefill on every single token from then on
	SlideContext bool

	// OnToken, when set, is called with each token as it is produced, before
	// the next forward pass runs. nil means nothing happens, which is what
	// every existing caller gets.
	//
	// this is what lets the repl print as it goes instead of sitting silent
	// for two seconds. note the callback receives token IDS, not text -> byte
	// level BPE means a single token can be a fragment of a utf-8 sequence,
	// so whoever is printing has to buffer bytes and only emit complete
	// characters. see the repl for how
	OnToken func(tokenID int)
}

func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		Temperature:       1.0,
		TopK:              40,
		RepetitionPenalty: 1.0,
	}
}

// slideContext throws away the oldest half of the context and rebuilds the KV
// caches from the rest, so generation can keep going past maxSeqLen.
//
// the rebuild is the whole trick. positional embeddings are added at the input
// and are therefore already inside every cached key and value, so the cache
// cannot simply be trimmed -> the retained entries would keep their old
// absolute positions while new tokens got numbered from the shortened length,
// putting two tokens at the same position. re-running the forward pass over
// the retained tokens with an empty cache renumbers them from zero.
//
// returns the logits for the last retained token, which is what the sampling
// loop needs to carry on
func (m *GPTModel) slideContext(tokens []int, caches []*KvCache) (*Matrix, error) {
	keep := m.maxSeqLen / 2
	if keep < 1 {
		keep = 1
	}
	if keep > len(tokens) {
		keep = len(tokens)
	}

	retained := tokens[len(tokens)-keep:]

	for i := range caches {
		caches[i] = &KvCache{}
	}

	return m.ForwardWithCache(retained, caches)
}

// isStopToken reports whether generation should end on this token
func (cfg GenerateConfig) isStopToken(tokenID int) bool {
	for _, stop := range cfg.StopTokenIDs {
		if tokenID == stop {
			return true
		}
	}
	return false
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

		if cfg.OnToken != nil {
			cfg.OnToken(nextToken)
		}

		// the model said it is done -> keep the token, it is part of the
		// output, but stop generating past it
		if cfg.isStopToken(nextToken) {
			break
		}

		// token caching
		if caches[0].key != nil && caches[0].key.rows >= m.maxSeqLen {
			if !cfg.SlideContext {
				break
			}

			var err error
			logits, err = m.slideContext(currentTokens, caches)
			if err != nil {
				return nil, err
			}
			continue
		}

		logits, err = m.ForwardWithCache([]int{nextToken}, caches)
		if err != nil {
			return nil, err
		}
	}

	return currentTokens, nil
}
