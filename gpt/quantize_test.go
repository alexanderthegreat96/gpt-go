package gpt

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizeRoundTripStaysWithinTheInt8Step(t *testing.T) {
	// int8 with a per column scale means each column is represented in 127
	// steps of its own maximum, so the worst reconstruction error for a
	// column is half a step -> maxAbs/254. anything worse than that means the
	// scale or the rounding is wrong
	rng := rand.New(rand.NewSource(1))

	rows, cols := 64, 16
	original := NewMatrix(rows, cols)
	for j := 0; j < cols; j++ {
		// give each column a wildly different magnitude, which is the whole
		// reason for scaling per column rather than per tensor
		columnScale := math.Pow(10, float64(j%5)-2)
		for i := 0; i < rows; i++ {
			original.data[i*cols+j] = float32((rng.Float64()*2 - 1) * columnScale)
		}
	}

	quantized := NewMatrix(rows, cols)
	copy(quantized.data, original.data)

	if err := quantized.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !quantized.IsQuantized() {
		t.Fatalf("matrix does not report itself as quantized")
	}
	if quantized.data != nil {
		t.Errorf("the float32 data should have been released, that is the point")
	}

	restored, err := quantized.Dequantize()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// per column maxima, to know what half a step is worth
	maxima := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if v := math.Abs(float64(original.data[i*cols+j])); v > maxima[j] {
				maxima[j] = v
			}
		}
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			diff := math.Abs(float64(original.data[idx] - restored.data[idx]))
			// half a step, with a little slack for float32 rounding of the
			// scale itself
			budget := maxima[j]/254 + maxima[j]*1e-6
			if diff > budget {
				t.Fatalf("element (%d,%d): original %g, restored %g, error %g exceeds the half step budget %g",
					i, j, original.data[idx], restored.data[idx], diff, budget)
			}
		}
	}
}

func TestQuantizedMatMulAgreesWithFloat32(t *testing.T) {
	rng := rand.New(rand.NewSource(2))

	rows, inner, cols := 5, 128, 32

	x := NewMatrix(rows, inner)
	for i := range x.data {
		x.data[i] = float32(rng.NormFloat64())
	}

	weights := NewMatrix(inner, cols)
	for i := range weights.data {
		weights.data[i] = float32(rng.NormFloat64() * 0.1)
	}

	exact, err := x.MatrixMul(weights)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	quantized := NewMatrix(inner, cols)
	copy(quantized.data, weights.data)
	if err := quantized.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// the same call site, dispatching to the int8 kernel because the right
	// hand operand is quantized
	approx, err := x.MatrixMul(quantized)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if approx.rows != exact.rows || approx.cols != exact.cols {
		t.Fatalf("expected %dx%d, got %dx%d", exact.rows, exact.cols, approx.rows, approx.cols)
	}

	// errors across a 128 long accumulation partially cancel, so the result
	// should be well inside a percent relative to the output magnitude
	var maxAbs, maxDiff float64
	for i := range exact.data {
		if v := math.Abs(float64(exact.data[i])); v > maxAbs {
			maxAbs = v
		}
		if d := math.Abs(float64(exact.data[i] - approx.data[i])); d > maxDiff {
			maxDiff = d
		}
	}

	if relative := maxDiff / maxAbs; relative > 0.02 {
		t.Errorf("int8 matmul is %0.4f off relative to the output scale, expected under 0.02", relative)
	}
}

func TestQuantizedMatMulMatchesDequantizedMatMul(t *testing.T) {
	// the int8 kernel should be arithmetically identical to dequantizing and
	// doing a normal float32 matmul, give or take float32 ordering. this is
	// what proves the per column scale is being applied in the right place
	rng := rand.New(rand.NewSource(3))

	x := NewMatrix(3, 24)
	for i := range x.data {
		x.data[i] = float32(rng.NormFloat64())
	}

	weights := NewMatrix(24, 8)
	for i := range weights.data {
		weights.data[i] = float32(rng.NormFloat64())
	}

	quantized := NewMatrix(24, 8)
	copy(quantized.data, weights.data)
	if err := quantized.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	viaKernel, err := x.MatrixMul(quantized)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dequantized, err := quantized.Dequantize()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	viaFloat, err := x.MatrixMul(dequantized)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i := range viaFloat.data {
		diff := math.Abs(float64(viaKernel.data[i] - viaFloat.data[i]))
		scale := math.Max(math.Abs(float64(viaFloat.data[i])), 1)
		if diff/scale > 1e-5 {
			t.Errorf("element %d: kernel gave %g, dequantize-then-multiply gave %g",
				i, viaKernel.data[i], viaFloat.data[i])
		}
	}
}

func TestQuantizeHandlesZeroColumns(t *testing.T) {
	// an all-zero column has no maximum to scale by. it must come back as
	// zeros rather than NaN from a division by zero
	m, err := NewMatrixFromData(2, 3, []float32{
		1, 0, -2,
		3, 0, 4,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := m.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	restored, err := m.Dequantize()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, v := range restored.data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("element %d came back NaN", i)
		}
	}
	if restored.data[1] != 0 || restored.data[4] != 0 {
		t.Errorf("the zero column should still be zero, got %g and %g", restored.data[1], restored.data[4])
	}
}

func TestQuantizeRefusesWhatItShould(t *testing.T) {
	// a matrix carrying gradients is being trained, and quantizing it would
	// throw away the weights the optimizer needs
	training := NewMatrix(2, 2)
	training.ensureGrad()
	if err := training.Quantize(); err == nil {
		t.Errorf("expected an error when quantizing a matrix with gradients")
	}

	// quantizing twice is a no-op rather than an error, so repeated calls are
	// harmless
	fine := NewMatrix(2, 2)
	fine.data[0] = 1
	if err := fine.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := fine.Quantize(); err != nil {
		t.Errorf("quantizing an already quantized matrix should be a no-op, got %v", err)
	}

	// and there is nothing to dequantize on a plain float32 matrix
	plain := NewMatrix(2, 2)
	if _, err := plain.Dequantize(); err == nil {
		t.Errorf("expected an error dequantizing a matrix that was never quantized")
	}
}

func TestQuantizedMatrixRefusesTrainingAndSaving(t *testing.T) {
	// the guards that stop a quantized model being used as if it were float32
	model, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads, Quantize: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !model.blocks[0].attention.wq.IsQuantized() {
		t.Fatalf("expected the loader to have quantized wq")
	}

	// backward would nil dereference on the released float32 data, so it has
	// to be refused outright
	x := NewMatrix(2, testDModel)
	dC := NewMatrix(2, testDModel)
	if _, _, err := x.MatrixMulBackward(model.blocks[0].attention.wq, dC); err == nil {
		t.Errorf("expected MatrixMulBackward to refuse a quantized weight")
	}

	// and our json format has nowhere to put int8 weights, so saving must
	// fail loudly rather than writing a file full of nulls
	if err := SaveModel(model, t.TempDir()+"/quantized.json"); err == nil {
		t.Errorf("expected SaveModel to refuse a quantized model")
	}
}

func TestLoadTimeQuantizationMatchesPostHoc(t *testing.T) {
	// quantizing during the load exists for peak memory, not for different
	// numbers -> both routes have to produce the same weights
	duringLoad, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads, Quantize: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	afterLoad, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := afterLoad.QuantizeForInference(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokens := []int{1, 2, 3}

	a, err := duringLoad.Forward(tokens)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	b, err := afterLoad.Forward(tokens)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i := range a.data {
		if a.data[i] != b.data[i] {
			t.Fatalf("logit %d differs between load-time and post-hoc quantization: %g vs %g",
				i, a.data[i], b.data[i])
		}
	}

	// and both should report the same saving
	if duringLoad.QuantizedBytesSaved() != afterLoad.QuantizedBytesSaved() {
		t.Errorf("saving differs: %d vs %d", duringLoad.QuantizedBytesSaved(), afterLoad.QuantizedBytesSaved())
	}
}

func TestQuantizedModelStaysCloseToFloat32(t *testing.T) {
	// end to end: the same model, with and without int8, should produce
	// logits that agree to within about a percent
	exact, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	approx, err := LoadGPT2(buildSyntheticGPT2(t, ""), GPT2Config{NumHeads: testNumHeads, Quantize: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tokens := []int{1, 2, 3, 4}

	exactLogits, err := exact.Forward(tokens)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	approxLogits, err := approx.Forward(tokens)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var maxAbs, maxDiff float64
	for i := range exactLogits.data {
		if v := math.Abs(float64(exactLogits.data[i])); v > maxAbs {
			maxAbs = v
		}
		if d := math.Abs(float64(exactLogits.data[i] - approxLogits.data[i])); d > maxDiff {
			maxDiff = d
		}
	}

	if maxAbs == 0 {
		t.Fatalf("the float32 model produced all-zero logits, the fixture is broken")
	}
	if relative := maxDiff / maxAbs; relative > 0.05 {
		t.Errorf("int8 logits are %0.4f off relative to the float32 scale, expected under 0.05", relative)
	}

	// the embeddings and layernorms must NOT have been quantized -> they are
	// read element-wise, not multiplied
	if approx.tokenEmbed.weights.IsQuantized() {
		t.Errorf("the token embeddings were quantized, but they are a lookup table")
	}
	if approx.finalLnGamma.IsQuantized() {
		t.Errorf("a layernorm parameter was quantized, but it is read per element")
	}
	if approx.blocks[0].attention.bq != nil && approx.blocks[0].attention.bq.IsQuantized() {
		t.Errorf("a bias was quantized, but biases are added rather than multiplied")
	}
}

func TestQuantizedBytesSavedMath(t *testing.T) {
	m := NewMatrix(100, 10)
	m.data[0] = 1

	plain := NewMatrix(100, 10)
	plain.data[0] = 1

	if err := m.Quantize(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 1000 elements: 4000 bytes as float32, 1000 as int8 plus 10 scales at
	// 4 bytes each
	model := &GPTModel{lmHead: m}
	expected := 4000 - (1000 + 40)
	if got := model.QuantizedBytesSaved(); got != expected {
		t.Errorf("expected a saving of %d bytes, got %d", expected, got)
	}

	// nothing quantized means nothing saved
	if got := (&GPTModel{lmHead: plain}).QuantizedBytesSaved(); got != 0 {
		t.Errorf("expected no saving for a float32 model, got %d", got)
	}
}
