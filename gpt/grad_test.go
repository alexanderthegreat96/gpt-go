package gpt

import (
	"math"
	"testing"
)

// gradient tests are here simply because we are lazy allocating
// gradients now
func meanLoss(t *testing.T, model *GPTModel, tokenIDs []int) float64 {
	t.Helper()

	inputIDs := tokenIDs[:len(tokenIDs)-1]
	targetIDs := tokenIDs[1:]

	logits, _, err := model.ForwardTrain(inputIDs)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	var total float64
	for i, target := range targetIDs {
		rowStart := i * model.vocabSize
		row, err := NewMatrixFromData(1, model.vocabSize, logits.data[rowStart:rowStart+model.vocabSize])
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		loss, _, err := CrossEntropyLoss(row, target)
		if err != nil {
			t.Fatalf("loss failed: %v", err)
		}
		total += float64(loss)
	}

	return total / float64(len(targetIDs))
}

// accumulateGradients runs forward and backward and leaves the gradients
// sitting on the parameters. deliberately no optimizer -> the weights must not
// move, or the finite differences we compare against would be measuring a
// different model
func accumulateGradients(t *testing.T, model *GPTModel, tokenIDs []int) {
	t.Helper()

	model.ZeroGrads()

	inputIDs := tokenIDs[:len(tokenIDs)-1]
	targetIDs := tokenIDs[1:]
	seqLen := len(inputIDs)

	logits, acts, err := model.ForwardTrain(inputIDs)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	dLogits := NewMatrix(logits.rows, logits.cols)
	for i, target := range targetIDs {
		rowStart := i * model.vocabSize
		row, err := NewMatrixFromData(1, model.vocabSize, logits.data[rowStart:rowStart+model.vocabSize])
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, gradRow, err := CrossEntropyLoss(row, target)
		if err != nil {
			t.Fatalf("loss failed: %v", err)
		}
		copy(dLogits.data[rowStart:rowStart+model.vocabSize], gradRow.data)
	}

	// same averaging TrainStep applies, so the gradients are of meanLoss
	inv := float32(1.0) / float32(seqLen)
	for i := range dLogits.data {
		dLogits.data[i] *= inv
	}

	if err := model.Backward(dLogits, acts); err != nil {
		t.Fatalf("backward pass failed: %v", err)
	}
}

func TestGradientsMatchNumericalGradient(t *testing.T) {
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3, 4}

	accumulateGradients(t, model, tokenIDs)

	probes := []struct {
		name  string
		param *Matrix
		index int
	}{
		{"block0 attention wq", model.blocks[0].attention.wq, 0},
		{"block0 attention bq", model.blocks[0].attention.bq, 0},
		{"block0 attention wo", model.blocks[0].attention.wo, 3},
		{"block0 attention bo", model.blocks[0].attention.bo, 1},
		{"block1 ffn w1", model.blocks[1].ffn.w1, 0},
		{"block1 ffn b1", model.blocks[1].ffn.b1, 2},
		{"block1 ffn w2", model.blocks[1].ffn.w2, 5},
		{"block1 ln1 gamma", model.blocks[1].ln1Gamma, 0},
		{"block1 ln2 beta", model.blocks[1].ln2Beta, 1},
		{"final ln gamma", model.finalLnGamma, 0},
		{"lm head", model.lmHead, 0},
		{"token embedding row 1", model.tokenEmbed.weights, 1 * testDModel},
		{"positional embedding", model.positionEmbed.weights, 0},
	}

	const h = 1e-2

	for _, probe := range probes {
		if probe.param == nil {
			t.Errorf("%s: parameter is nil, cannot check its gradient", probe.name)
			continue
		}
		if probe.param.grad == nil {
			t.Errorf("%s: no gradient was allocated at all, AccumulateGrad is silently dropping it", probe.name)
			continue
		}

		analytic := float64(probe.param.grad[probe.index])
		original := probe.param.data[probe.index]

		probe.param.data[probe.index] = original + h
		lossPlus := meanLoss(t, model, tokenIDs)

		probe.param.data[probe.index] = original - h
		lossMinus := meanLoss(t, model, tokenIDs)

		probe.param.data[probe.index] = original

		numerical := (lossPlus - lossMinus) / (2 * h)

		// accept either a tight absolute match or 5% relative, which is about
		// what float32 finite differences can promise
		absDiff := math.Abs(analytic - numerical)
		scale := math.Max(math.Abs(analytic), math.Abs(numerical))
		if absDiff > 2e-2 && absDiff > 0.05*scale {
			t.Errorf("%s: analytic gradient %.6f but numerical is %.6f (diff %.6f)",
				probe.name, analytic, numerical, absDiff)
		}
	}
}

func TestFreshlyLoadedModelHasNoGradientBuffers(t *testing.T) {
	// this is the whole point of the change -> a model we only ever run
	// forward should not be carrying a gradient buffer for every weight.
	// on gpt2-medium those buffers were about 1.6GB
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, param := range model.GetTrainableParameters() {
		if param == nil {
			continue
		}
		if param.grad != nil {
			t.Errorf("parameter %d already has a gradient buffer of %d floats after a plain load", i, len(param.grad))
		}
	}

	// and running inference must not change that
	if _, err := model.Forward([]int{1, 2, 3}); err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	for i, param := range model.GetTrainableParameters() {
		if param != nil && param.grad != nil {
			t.Errorf("parameter %d grew a gradient buffer during a forward pass", i)
		}
	}
}

func TestZeroGradsOnAFreshModelIsSafeAndAllocatesNothing(t *testing.T) {
	// TrainStep calls ZeroGrads before anything else, so this runs against
	// nil buffers on the very first step of every training run
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	model.ZeroGrads()

	for i, param := range model.GetTrainableParameters() {
		if param != nil && param.grad != nil {
			t.Errorf("parameter %d was given a gradient buffer by ZeroGrads, which should be a no-op", i)
		}
	}
}

func TestTrainStepAllocatesGradientsAndMovesWeights(t *testing.T) {
	// the other half of the contract: training has to work exactly as it did
	// before, just allocating a moment later
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	before := make([]float32, len(model.blocks[0].attention.wq.data))
	copy(before, model.blocks[0].attention.wq.data)

	optimizer := NewAdamW(0.01, 0.9, 0.999, 1e-8, 0.01)

	loss, err := model.TrainStep([]int{1, 2, 3, 4}, optimizer)
	if err != nil {
		t.Fatalf("train step failed: %v", err)
	}
	if loss <= 0 || loss != loss {
		t.Errorf("expected a positive finite loss, got %f", loss)
	}

	// gradients should exist now
	if model.blocks[0].attention.wq.grad == nil {
		t.Fatalf("wq has no gradient buffer after a training step")
	}
	if model.blocks[0].attention.bq.grad == nil {
		t.Errorf("the attention bias has no gradient buffer after a training step")
	}

	// and the weights have to have actually moved. a zeroed gradient would
	// leave them exactly where they started, which is the failure mode that
	// started all of this
	moved := false
	for i := range before {
		if before[i] != model.blocks[0].attention.wq.data[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Errorf("wq did not change after a training step, gradients are being dropped")
	}
}

func TestLossDecreasesOverSeveralSteps(t *testing.T) {
	// an end to end sanity check. gradients can be individually plausible and
	// still not add up to learning, so overfit four tokens and watch the loss
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokenIDs := []int{1, 2, 3, 4}
	optimizer := NewAdamW(0.05, 0.9, 0.999, 1e-8, 0.0)

	first, err := model.TrainStep(tokenIDs, optimizer)
	if err != nil {
		t.Fatalf("train step failed: %v", err)
	}

	var last float32
	for step := 0; step < 20; step++ {
		last, err = model.TrainStep(tokenIDs, optimizer)
		if err != nil {
			t.Fatalf("train step %d failed: %v", step, err)
		}
	}

	if last >= first {
		t.Errorf("loss did not come down over 20 steps: started at %.4f, ended at %.4f", first, last)
	}
}
