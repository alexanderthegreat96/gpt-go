package gpt

import (
	"fmt"
	"math"
	"sync"
)

type KvCache struct {
	key   *Matrix
	value *Matrix
}

func NewKvCache(key *Matrix, value *Matrix) *KvCache {
	return &KvCache{
		key:   key,
		value: value,
	}
}

func (c *KvCache) Clear() {
	c.key = nil
	c.value = nil
}

type CausalAttention struct {
	dModel   int
	numHeads int
	headDim  int

	wq *Matrix
	wk *Matrix
	wv *Matrix
	wo *Matrix
}

func NewCausalAttention(dmodel, numHeads int, scale float32) (*CausalAttention, error) {
	if dmodel%numHeads != 0 {
		return nil, fmt.Errorf("dModel (%d) must be divisible by numHeads (%d)", dmodel, numHeads)
	}

	headDim := dmodel / numHeads

	wq := NewMatrix(dmodel, dmodel)
	wk := NewMatrix(dmodel, dmodel)
	wv := NewMatrix(dmodel, dmodel)
	wo := NewMatrix(dmodel, dmodel)

	wq.Randomize(scale)
	wk.Randomize(scale)
	wv.Randomize(scale)
	wo.Randomize(scale)

	return &CausalAttention{
		dModel:   dmodel,
		numHeads: numHeads,
		headDim:  headDim,
		wq:       wq,
		wk:       wk,
		wv:       wv,
		wo:       wo,
	}, nil
}

func (ca *CausalAttention) Project(input *Matrix, weight *Matrix) (*Matrix, error) {
	return input.MatrixMul(weight)
}

func (ca *CausalAttention) Forward(input *Matrix) (*Matrix, error) {
	seqLen := input.rows

	q, err := input.MatrixMul(ca.wq)
	if err != nil {
		return nil, err
	}

	k, err := input.MatrixMul(ca.wk)
	if err != nil {
		return nil, err
	}

	v, err := input.MatrixMul(ca.wv)
	if err != nil {
		return nil, err
	}

	scale := float32(1.0 / math.Sqrt(float64(ca.headDim)))
	headOutputs := NewMatrix(seqLen, ca.dModel)

	var wg sync.WaitGroup

	for h := 0; h < ca.numHeads; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()

			scores := NewMatrix(seqLen, seqLen)

			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					if j > i {
						scores.data[i*seqLen+j] = -1e9
						continue
					}

					var dot float32 = 0
					for d := 0; d < ca.headDim; d++ {
						qIdx := i*ca.dModel + (h*ca.headDim + d)
						kIdx := j*ca.dModel + (h*ca.headDim + d)
						dot += q.data[qIdx] * k.data[kIdx]
					}
					scores.data[i*seqLen+j] = dot * scale
				}
			}

			attentionProbs := scores.Softmax()

			for i := 0; i < seqLen; i++ {
				for d := 0; d < ca.headDim; d++ {
					var val float32 = 0
					for j := 0; j < seqLen; j++ {
						prob := attentionProbs.data[i*seqLen+j]
						vIdx := j*ca.dModel + (h*ca.headDim + d)
						val += prob * v.data[vIdx]
					}
					headOutputs.data[i*ca.dModel+(h*ca.headDim+d)] = val
				}
			}
		}(h)
	}

	wg.Wait()

	output, err := headOutputs.MatrixMul(ca.wo)
	if err != nil {
		return nil, err
	}

	return output, nil
}

type AttentionActivations struct {
	input       *Matrix
	q, k, v     *Matrix
	probs       []*Matrix
	headOutputs *Matrix
}

func (ca *CausalAttention) ForwardSave(input *Matrix) (*Matrix, *AttentionActivations, error) {
	seqLen := input.rows

	q, err := input.MatrixMul(ca.wq)
	if err != nil {
		return nil, nil, err
	}

	k, err := input.MatrixMul(ca.wk)
	if err != nil {
		return nil, nil, err
	}

	v, err := input.MatrixMul(ca.wv)
	if err != nil {
		return nil, nil, err
	}

	scale := float32(1.0 / math.Sqrt(float64(ca.headDim)))
	headOutputs := NewMatrix(seqLen, ca.dModel)
	probs := make([]*Matrix, ca.numHeads)

	for h := 0; h < ca.numHeads; h++ {
		scores := NewMatrix(seqLen, seqLen)

		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if j > i {
					scores.data[i*seqLen+j] = -1e9
					continue
				}
				var dot float32
				for d := 0; d < ca.headDim; d++ {
					dot += q.data[i*ca.dModel+h*ca.headDim+d] * k.data[j*ca.dModel+h*ca.headDim+d]
				}
				scores.data[i*seqLen+j] = dot * scale
			}
		}

		attentionProbs := scores.Softmax()
		probs[h] = attentionProbs

		for i := 0; i < seqLen; i++ {
			for d := 0; d < ca.headDim; d++ {
				var val float32
				for j := 0; j < seqLen; j++ {
					val += attentionProbs.data[i*seqLen+j] * v.data[j*ca.dModel+h*ca.headDim+d]
				}
				headOutputs.data[i*ca.dModel+h*ca.headDim+d] = val
			}
		}
	}

	output, err := headOutputs.MatrixMul(ca.wo)
	if err != nil {
		return nil, nil, err
	}

	acts := &AttentionActivations{
		input:       input,
		q:           q,
		k:           k,
		v:           v,
		probs:       probs,
		headOutputs: headOutputs,
	}

	return output, acts, nil
}

func (ca *CausalAttention) Backward(acts *AttentionActivations, dOut *Matrix) (*Matrix, error) {
	seqLen := acts.input.rows
	scale := float32(1.0 / math.Sqrt(float64(ca.headDim)))

	dHeadOutputs, dWo, err := acts.headOutputs.MatrixMulBackward(ca.wo, dOut)
	if err != nil {
		return nil, err
	}
	ca.wo.AccumulateGrad(dWo)

	dQ := NewMatrix(seqLen, ca.dModel)
	dK := NewMatrix(seqLen, ca.dModel)
	dV := NewMatrix(seqLen, ca.dModel)

	for h := 0; h < ca.numHeads; h++ {
		probsH := acts.probs[h]
		dProbs := NewMatrix(seqLen, seqLen)

		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				var dot float32
				for d := 0; d < ca.headDim; d++ {
					dot += dHeadOutputs.data[i*ca.dModel+h*ca.headDim+d] * acts.v.data[j*ca.dModel+h*ca.headDim+d]
				}
				dProbs.data[i*seqLen+j] = dot
			}
		}

		for j := 0; j < seqLen; j++ {
			for d := 0; d < ca.headDim; d++ {
				var sum float32
				for i := 0; i < seqLen; i++ {
					sum += probsH.data[i*seqLen+j] * dHeadOutputs.data[i*ca.dModel+h*ca.headDim+d]
				}
				dV.data[j*ca.dModel+h*ca.headDim+d] += sum
			}
		}

		dScores := SoftmaxBackward(probsH, dProbs)
		for i := 0; i < seqLen; i++ {
			for j := i + 1; j < seqLen; j++ {
				dScores.data[i*seqLen+j] = 0
			}
		}

		for i := 0; i < seqLen; i++ {
			for d := 0; d < ca.headDim; d++ {
				var sumQ float32
				for j := 0; j < seqLen; j++ {
					sumQ += dScores.data[i*seqLen+j] * acts.k.data[j*ca.dModel+h*ca.headDim+d]
				}
				dQ.data[i*ca.dModel+h*ca.headDim+d] += sumQ * scale
			}
		}

		for j := 0; j < seqLen; j++ {
			for d := 0; d < ca.headDim; d++ {
				var sumK float32
				for i := 0; i < seqLen; i++ {
					sumK += dScores.data[i*seqLen+j] * acts.q.data[i*ca.dModel+h*ca.headDim+d]
				}
				dK.data[j*ca.dModel+h*ca.headDim+d] += sumK * scale
			}
		}
	}

	dInputQ, dWq, err := acts.input.MatrixMulBackward(ca.wq, dQ)
	if err != nil {
		return nil, err
	}
	ca.wq.AccumulateGrad(dWq)

	dInputK, dWk, err := acts.input.MatrixMulBackward(ca.wk, dK)
	if err != nil {
		return nil, err
	}
	ca.wk.AccumulateGrad(dWk)

	dInputV, dWv, err := acts.input.MatrixMulBackward(ca.wv, dV)
	if err != nil {
		return nil, err
	}
	ca.wv.AccumulateGrad(dWv)

	dInput, err := dInputQ.MatrixAdd(dInputK)
	if err != nil {
		return nil, err
	}
	dInput, err = dInput.MatrixAdd(dInputV)
	if err != nil {
		return nil, err
	}

	return dInput, nil
}

func (ca *CausalAttention) ForwardWithCache(input *Matrix, cache *KvCache) (*Matrix, error) {
	currentSeqLen := input.rows

	q, err := input.MatrixMul(ca.wq)
	if err != nil {
		return nil, err
	}
	newK, err := input.MatrixMul(ca.wk)
	if err != nil {
		return nil, err
	}
	newV, err := input.MatrixMul(ca.wv)
	if err != nil {
		return nil, err
	}

	var k, v *Matrix
	if cache != nil {
		k, err = cache.key.ConcatRows(newK)
		if err != nil {
			return nil, err
		}
		v, err = cache.value.ConcatRows(newV)
		if err != nil {
			return nil, err
		}
		cache.key = k
		cache.value = v
	} else {
		k = newK
		v = newV
	}

	totalSeqLen := k.rows
	scale := float32(1.0 / math.Sqrt(float64(ca.headDim)))
	headOutputs := NewMatrix(currentSeqLen, ca.dModel)

	var wg sync.WaitGroup
	for h := 0; h < ca.numHeads; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()

			scores := NewMatrix(currentSeqLen, totalSeqLen)

			for i := 0; i < currentSeqLen; i++ {
				absoluteQueryIdx := (totalSeqLen - currentSeqLen) + i
				for j := 0; j < totalSeqLen; j++ {
					if j > absoluteQueryIdx {
						scores.data[i*totalSeqLen+j] = -1e9
						continue
					}

					var dot float32 = 0
					for d := 0; d < ca.headDim; d++ {
						qIdx := i*ca.dModel + (h*ca.headDim + d)
						kIdx := j*ca.dModel + (h*ca.headDim + d)
						dot += q.data[qIdx] * k.data[kIdx]
					}
					scores.data[i*totalSeqLen+j] = dot * scale
				}
			}

			attentionProbs := scores.Softmax()

			for i := 0; i < currentSeqLen; i++ {
				for d := 0; d < ca.headDim; d++ {
					var val float32 = 0
					for j := 0; j < totalSeqLen; j++ {
						prob := attentionProbs.data[i*totalSeqLen+j]
						vIdx := j*ca.dModel + (h*ca.headDim + d)
						val += prob * v.data[vIdx]
					}
					headOutputs.data[i*ca.dModel+(h*ca.headDim+d)] = val
				}
			}
		}(h)
	}
	wg.Wait()

	output, err := headOutputs.MatrixMul(ca.wo)
	if err != nil {
		return nil, err
	}

	return output, nil
}
