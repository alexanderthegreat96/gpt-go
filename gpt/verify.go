package gpt

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Reference Verification
// this file exists because of one nasty property of loading somebody else's
// weights: getting it wrong does not produce an error.
//
// transpose one matrix, use the erf GELU where the tanh one belongs, swap two
// heads, forget a bias -> the model still runs, still emits grammatical
// english, and still looks exactly like a model that merely needs more
// training. you can lose a week to that.
//
// so we don't debug by reading output. we dump the intermediate activations out
// of a known-good pytorch gpt-2 for a fixed prompt (scripts/dump_gpt2_reference.py),
// then run the same tokens through our forward pass and compare tensor by
// tensor. instead of "the output is garbage" you get "block 3 attention
// diverges, everything before it matches to 6 decimals", which is a bug you can
// fix in an afternoon

// the activation points we compare, in the order data flows through the model.
// these strings have to match what the python dumper writes, so if you add one
// it goes in both places
const (
	refEmbed  = "embed"  // token + positional embeddings, before block 0
	refLnF    = "ln_f"   // the final layernorm
	refLogits = "logits" // what comes out of the lm head
)

// and the per-block ones, formatted with the block index
const (
	refBlockLn1  = "h.%d.ln_1" // after the first layernorm
	refBlockAttn = "h.%d.attn" // attention output, before the residual add
	refBlockRes1 = "h.%d.res1" // after the first residual add
	refBlockLn2  = "h.%d.ln_2" // after the second layernorm
	refBlockMLP  = "h.%d.mlp"  // mlp output, before the residual add
	refBlockOut  = "h.%d.out"  // the block's final output
)

// Tolerance is what we will accept as "the same tensor".
//
// a single absolute threshold does not work here, and finding that out cost a
// round of chasing a bug that did not exist. gpt-2's activations span five
// orders of magnitude: the embedding sum peaks around 2.5 while the mlp output
// in block 3 reaches 3308, because of the outlier features the architecture is
// known for. float32 has about 7 digits, so an element of magnitude 3000
// carries an absolute rounding error of ~1e-3 all by itself, before you
// accumulate it across a 4096 wide dot product.
//
// so we judge relatively instead, on two numbers that fail for different
// reasons:
//
//   - MaxRel catches a few badly wrong elements. big values, small mean error
//   - MeanRel catches an error smeared across everything, which is what a
//     wrong activation function or a missing bias actually looks like
//
// Abs stays as an escape hatch for tensors whose magnitude is so small that a
// relative figure is meaningless
type Tolerance struct {
	Abs     float32
	MaxRel  float32
	MeanRel float32
}

// DefaultTolerance is calibrated against a real gpt2-medium run: the worst
// relative error observed across all 147 activation points was 2.9e-5 for the
// max and about 1e-5 for the mean. these thresholds sit roughly an order of
// magnitude above that noise floor, and roughly an order of magnitude below
// what an actual bug produces -> swapping the tanh GELU for the erf one moves
// the mean relative error to ~1e-3
func DefaultTolerance() Tolerance {
	return Tolerance{Abs: 1e-3, MaxRel: 1e-3, MeanRel: 1e-4}
}

// AbsoluteTolerance accepts only an exact-ish absolute match. the unit tests
// compare a model against its own activations, where anything but a tiny
// absolute difference means something is actually broken
func AbsoluteTolerance(abs float32) Tolerance {
	return Tolerance{Abs: abs}
}

// TensorDiff is the verdict on one activation point
type TensorDiff struct {
	Name string

	// Status is "ok" when we compared it, "missing" when the reference file
	// doesn't carry that tensor, and "shape" when the two disagree on size
	Status string

	MaxAbsDiff  float32
	MeanAbsDiff float32

	// the scale of the reference tensor, which is what makes the absolute
	// numbers above interpretable
	RefMaxAbs  float32
	RefMeanAbs float32

	MaxRelDiff  float32
	MeanRelDiff float32

	RefRows, RefCols int
	GotRows, GotCols int
}

// Passed reports whether this tensor is within tolerance.
// a missing tensor is not a failure -> the reference dumper may simply not
// have been asked for it. a shape mismatch always is
func (d TensorDiff) Passed(tol Tolerance) bool {
	switch d.Status {
	case "missing":
		return true
	case "shape":
		return false
	}

	// an absolutely tiny error is fine regardless of scale
	if d.MaxAbsDiff <= tol.Abs {
		return true
	}

	// otherwise both relative figures have to hold. requiring both means a
	// handful of rounding outliers in a large tensor passes, while an error
	// spread across every element does not
	return d.MaxRelDiff <= tol.MaxRel && d.MeanRelDiff <= tol.MeanRel
}

// CollectActivations runs a full forward pass and hands back every point we
// know how to compare, along with the order to walk them in.
//
// we get almost all of this for free: ForwardTrain already stashes the
// per-block intermediates because the backward pass needs them, so there are
// no extra hooks to maintain and no second forward implementation that can
// drift away from the real one
func CollectActivations(m *GPTModel, tokenIDs []int) ([]string, map[string]*Matrix, error) {
	if len(tokenIDs) == 0 {
		return nil, nil, fmt.Errorf("verify: no tokens to run")
	}
	if len(tokenIDs) > m.maxSeqLen {
		return nil, nil, fmt.Errorf("verify: %d tokens exceeds the model context of %d", len(tokenIDs), m.maxSeqLen)
	}

	logits, acts, err := m.ForwardTrain(tokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("verify: forward pass failed: %v", err)
	}

	order := make([]string, 0, 6*m.numLayers+3)
	points := make(map[string]*Matrix, 6*m.numLayers+3)

	add := func(name string, value *Matrix) {
		order = append(order, name)
		points[name] = value
	}

	// the embedding sum is whatever went into block 0
	if len(acts.blockActs) > 0 {
		add(refEmbed, acts.blockActs[0].blockInput)
	}

	for i, blockActs := range acts.blockActs {
		add(fmt.Sprintf(refBlockLn1, i), blockActs.norm1Out)
		add(fmt.Sprintf(refBlockAttn, i), blockActs.attnOut)
		add(fmt.Sprintf(refBlockRes1, i), blockActs.res1)
		add(fmt.Sprintf(refBlockLn2, i), blockActs.norm2Out)
		add(fmt.Sprintf(refBlockMLP, i), blockActs.ffnOut)
		add(fmt.Sprintf(refBlockOut, i), blockActs.output)
	}

	add(refLnF, acts.finalNormed)
	add(refLogits, logits)

	return order, points, nil
}

// VerifyAgainstReference is the whole point of this file.
//
// it reads a reference dump, pulls the token ids out of its metadata so both
// sides are definitely running the same prompt, runs our forward pass and
// compares every activation point the dump contains. the diffs come back in
// network order, so the FIRST failure in the list is the one to go fix -> every
// divergence after it is just that error flowing downstream
func VerifyAgainstReference(m *GPTModel, referencePath string, tol Tolerance) ([]TensorDiff, []int, bool, error) {
	ref, err := OpenSafeTensors(referencePath)
	if err != nil {
		return nil, nil, false, err
	}
	defer ref.Close()

	tokenIDs, err := parseReferenceTokens(ref)
	if err != nil {
		return nil, nil, false, err
	}

	// before comparing anything, make sure the two files are even describing
	// the same model. otherwise you get a wall of shape errors that reads like
	// a transpose bug when really you just downloaded the wrong checkpoint
	if err := crossCheckReference(m, ref); err != nil {
		return nil, nil, false, err
	}

	order, points, err := CollectActivations(m, tokenIDs)
	if err != nil {
		return nil, nil, false, err
	}

	diffs := make([]TensorDiff, 0, len(order))
	allPassed := true

	for _, name := range order {
		got := points[name]
		diff := TensorDiff{
			Name:    name,
			GotRows: got.rows,
			GotCols: got.cols,
		}

		if !ref.Has(name) {
			diff.Status = "missing"
			diffs = append(diffs, diff)
			continue
		}

		expected, err := ref.Matrix(name)
		if err != nil {
			return nil, nil, false, fmt.Errorf("verify: reading reference tensor %q: %v", name, err)
		}

		diff.RefRows = expected.rows
		diff.RefCols = expected.cols

		if expected.rows != got.rows || expected.cols != got.cols {
			// almost always a transpose or a head-layout mistake
			diff.Status = "shape"
			diffs = append(diffs, diff)
			allPassed = false
			continue
		}

		diff.Status = "ok"
		diff.MaxAbsDiff, diff.MeanAbsDiff = compareTensors(expected, got)
		diff.RefMaxAbs, diff.RefMeanAbs = tensorScale(expected)

		// guard the division -> an all-zero reference tensor would otherwise
		// produce NaN and read as a failure
		if diff.RefMaxAbs > 0 {
			diff.MaxRelDiff = diff.MaxAbsDiff / diff.RefMaxAbs
		}
		if diff.RefMeanAbs > 0 {
			diff.MeanRelDiff = diff.MeanAbsDiff / diff.RefMeanAbs
		}

		if !diff.Passed(tol) {
			allPassed = false
		}

		diffs = append(diffs, diff)
	}

	return diffs, tokenIDs, allPassed, nil
}

// crossCheckReference refuses to compare a model against a reference that was
// clearly dumped from a different one.
//
// this check is here because of how badly the alternative reads. load
// gpt2-medium, compare it against a dump from gpt2, and every single tensor
// comes back as a shape mismatch -> which looks exactly like a transposed
// matrix or a botched qkv split, so that is what you go and debug. the real
// problem is that the two files are different models, and it is much better to
// say so in one sentence than to let somebody read 75 rows of noise
func crossCheckReference(m *GPTModel, ref *SafeTensors) error {
	_, dModel, numLayers, _, _ := m.Describe()

	// newer dumps state their shape outright in the metadata. older ones do
	// not, so we fall back to reading it off the tensors: the final layernorm
	// output is seqLen x dModel, and the block outputs give us the depth
	refLayers := metadataInt(ref, "n_layer")
	if refLayers == 0 {
		refLayers = countReferenceLayers(ref)
	}

	refDModel := metadataInt(ref, "n_embd")
	if refDModel == 0 {
		if shape, err := ref.Shape(refLnF); err == nil && len(shape) == 2 {
			refDModel = shape[1]
		}
	}

	if (refDModel == 0 || refDModel == dModel) && (refLayers == 0 || refLayers == numLayers) {
		return nil
	}

	// a partial dump is fine, so only complain about what we could actually
	// read out of it
	refDesc := "the reference"
	if name, ok := ref.Metadata["model"]; ok && name != "" {
		refDesc = fmt.Sprintf("the reference (dumped from %q)", name)
	}

	return fmt.Errorf(
		"verify: these two files describe different models.\n"+
			"  the weights are %d layers, %d wide%s\n"+
			"  %s is %s layers, %s wide%s\n"+
			"re-dump the reference against the same checkpoint you are loading, or load the\n"+
			"checkpoint the reference came from",
		numLayers, dModel, sizeSuffix(gpt2SizeName(dModel, numLayers)),
		refDesc, describeDim(refLayers), describeDim(refDModel),
		sizeSuffix(gpt2SizeName(refDModel, refLayers)),
	)
}

// metadataInt reads a numeric metadata value, returning 0 when it is absent or
// unparseable. absent is normal -> dumps written before we started recording
// the model shape simply do not have it
func metadataInt(ref *SafeTensors, key string) int {
	raw, ok := ref.Metadata[key]
	if !ok {
		return 0
	}
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0
	}
	return value
}

// countReferenceLayers works out how many blocks a dump covers by finding the
// highest h.N it carries. we go by the block output rather than counting keys,
// since a partial dump might only contain some points per block
func countReferenceLayers(ref *SafeTensors) int {
	count := 0
	for {
		if !ref.Has(fmt.Sprintf(refBlockOut, count)) {
			return count
		}
		count++
	}
}

// describeDim prints a dimension we could not determine as a question mark
// rather than a confusing zero
func describeDim(value int) string {
	if value == 0 {
		return "?"
	}
	return strconv.Itoa(value)
}

// sizeSuffix turns a recognized gpt-2 size into something we can append to a
// sentence, and stays quiet about anything it does not recognize
func sizeSuffix(name string) string {
	if name == "" {
		return ""
	}
	return " (looks like " + name + ")"
}

// compareTensors returns the largest and the average absolute difference
// between two identically shaped matrices.
//
// we care about both: the max catches a handful of badly wrong values that an
// average would bury, while the mean tells you whether something is subtly off
// everywhere (which is what a wrong activation function looks like) rather than
// catastrophically wrong in one spot
func compareTensors(expected, got *Matrix) (float32, float32) {
	var maxDiff float64
	var sumDiff float64

	for i := range expected.data {
		d := math.Abs(float64(expected.data[i]) - float64(got.data[i]))
		if d > maxDiff {
			maxDiff = d
		}
		sumDiff += d
	}

	mean := sumDiff / float64(len(expected.data))

	return float32(maxDiff), float32(mean)
}

// tensorScale returns the largest and the average absolute value in a tensor.
//
// this is what turns the diff numbers into something you can judge. "7e-3 is
// wrong" is only true if the values involved are small -> in block 3's mlp
// output they reach 3308, and at that magnitude 7e-3 is the eighth significant
// digit, which is simply where float32 runs out
func tensorScale(m *Matrix) (float32, float32) {
	var maxAbs float64
	var sumAbs float64

	for _, v := range m.data {
		a := math.Abs(float64(v))
		if a > maxAbs {
			maxAbs = a
		}
		sumAbs += a
	}

	return float32(maxAbs), float32(sumAbs / float64(len(m.data)))
}

// parseReferenceTokens digs the prompt token ids out of the dump's metadata.
//
// safetensors metadata values have to be strings, so the python side writes
// them comma separated. carrying the tokens inside the dump rather than
// re-tokenizing on our side is deliberate -> we don't have gpt-2's BPE yet, and
// even once we do, a tokenizer bug would otherwise show up as a model bug
func parseReferenceTokens(ref *SafeTensors) ([]int, error) {
	raw, ok := ref.Metadata["tokens"]
	if !ok {
		return nil, fmt.Errorf("verify: the reference dump has no \"tokens\" metadata, re-run the python dumper")
	}

	fields := strings.Split(raw, ",")
	tokenIDs := make([]int, 0, len(fields))

	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		id, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf("verify: token list contains %q which is not a number", field)
		}
		tokenIDs = append(tokenIDs, id)
	}

	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("verify: the reference dump carries an empty token list")
	}

	return tokenIDs, nil
}

// Shape is a small accessor so the cli can print matrix dimensions without
// living inside this package
func (m *Matrix) Shape() (int, int) {
	return m.rows, m.cols
}
