package gpt

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeTestSafeTensors builds a real safetensors file on disk so the reader is
// exercised end to end rather than against a mock. the layout is the same one
// the reader expects: 8 bytes of header length, the json header, then the raw
// tensor bytes with offsets relative to the start of that data section
func writeTestSafeTensors(t *testing.T, tensors map[string][]float32, shapes map[string][]int, metadata map[string]string) string {
	t.Helper()

	header := map[string]any{}
	var blob []byte
	offset := 0

	// map iteration order is random, so we sort the names to keep the file
	// deterministic between runs
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}

	for _, name := range names {
		values := tensors[name]
		start := offset
		for _, v := range values {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			blob = append(blob, buf[:]...)
			offset += 4
		}

		header[name] = map[string]any{
			"dtype":        "F32",
			"shape":        shapes[name],
			"data_offsets": []int{start, offset},
		}
	}

	if metadata != nil {
		header["__metadata__"] = metadata
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("could not marshal the test header: %v", err)
	}

	var file []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	file = append(file, lenBuf[:]...)
	file = append(file, headerBytes...)
	file = append(file, blob...)

	path := filepath.Join(t.TempDir(), "test.safetensors")
	if err := os.WriteFile(path, file, 0644); err != nil {
		t.Fatalf("could not write the test file: %v", err)
	}

	return path
}

func TestSafeTensorsRoundTrip(t *testing.T) {
	path := writeTestSafeTensors(t,
		map[string][]float32{
			"weight": {1, 2, 3, 4, 5, 6},
			"bias":   {0.5, -0.5},
		},
		map[string][]int{
			"weight": {3, 2},
			"bias":   {2},
		},
		map[string]string{"tokens": "15496,11,616"},
	)

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer st.Close()

	// a 2D tensor keeps its shape
	weight, err := st.Matrix("weight")
	if err != nil {
		t.Fatalf("unexpected error reading weight: %v", err)
	}
	if weight.rows != 3 || weight.cols != 2 {
		t.Errorf("expected weight to be 3x2, got %dx%d", weight.rows, weight.cols)
	}
	expected := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range expected {
		if weight.data[i] != v {
			t.Errorf("weight at index %d: expected %f, got %f", i, v, weight.data[i])
		}
	}

	// a 1D tensor becomes a single row, which is how every bias and layernorm
	// parameter reaches us
	bias, err := st.Matrix("bias")
	if err != nil {
		t.Fatalf("unexpected error reading bias: %v", err)
	}
	if bias.rows != 1 || bias.cols != 2 {
		t.Errorf("expected bias to be 1x2, got %dx%d", bias.rows, bias.cols)
	}

	if st.Metadata["tokens"] != "15496,11,616" {
		t.Errorf("metadata did not survive the round trip, got %q", st.Metadata["tokens"])
	}

	if st.Has("nope") {
		t.Errorf("Has reported a tensor that is not in the file")
	}
	if _, err := st.Matrix("nope"); err == nil {
		t.Errorf("expected an error for a missing tensor")
	}
}

func TestSafeTensorsRejectsTruncatedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "truncated.safetensors")

	// a header length that promises far more than the file actually contains.
	// this is the failure we most want caught early, since the alternative is
	// slicing past the end of the buffer later on
	var file []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], 9999)
	file = append(file, lenBuf[:]...)
	file = append(file, []byte("{}")...)

	if err := os.WriteFile(path, file, 0644); err != nil {
		t.Fatalf("could not write the test file: %v", err)
	}

	if _, err := OpenSafeTensors(path); err == nil {
		t.Errorf("expected an error for a header longer than the file")
	}
}

func TestHalfToFloat32(t *testing.T) {
	cases := []struct {
		name     string
		bits     uint16
		expected float32
	}{
		{"zero", 0x0000, 0},
		{"negative zero", 0x8000, 0},
		{"one", 0x3C00, 1},
		{"negative two", 0xC000, -2},
		{"half", 0x3800, 0.5},
		{"largest normal", 0x7BFF, 65504},
		// the smallest subnormal, which is the case the shifting loop exists for
		{"smallest subnormal", 0x0001, float32(math.Pow(2, -24))},
	}

	for _, c := range cases {
		got := halfToFloat32(c.bits)
		if math.Abs(float64(got-c.expected)) > 1e-9 {
			t.Errorf("%s: expected %v, got %v", c.name, c.expected, got)
		}
	}

	// infinity and NaN have to survive too, since a bad checkpoint will have
	// them and we would rather see them than a silently wrapped number
	if !math.IsInf(float64(halfToFloat32(0x7C00)), 1) {
		t.Errorf("expected positive infinity")
	}
	if !math.IsNaN(float64(halfToFloat32(0x7E00))) {
		t.Errorf("expected NaN")
	}
}

func TestSplitFusedQKV(t *testing.T) {
	// dModel of 2, so the fused matrix is 2x6 and every row holds
	// [ q0 q1 | k0 k1 | v0 v1 ]. getting this wrong by taking the first third
	// of the flat buffer instead of walking row by row is the exact mistake
	// this test is here to catch
	fused, err := NewMatrixFromData(2, 6, []float32{
		1, 2, 10, 20, 100, 200,
		3, 4, 30, 40, 300, 400,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	wq, wk, wv, err := splitFusedQKV(fused, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	checks := []struct {
		name     string
		got      *Matrix
		expected []float32
	}{
		{"wq", wq, []float32{1, 2, 3, 4}},
		{"wk", wk, []float32{10, 20, 30, 40}},
		{"wv", wv, []float32{100, 200, 300, 400}},
	}

	for _, c := range checks {
		if c.got.rows != 2 || c.got.cols != 2 {
			t.Errorf("%s: expected 2x2, got %dx%d", c.name, c.got.rows, c.got.cols)
			continue
		}
		for i, v := range c.expected {
			if c.got.data[i] != v {
				t.Errorf("%s at index %d: expected %f, got %f", c.name, i, v, c.got.data[i])
			}
		}
	}

	// a wrongly shaped c_attn should be refused rather than reinterpreted
	wrong, _ := NewMatrixFromData(2, 4, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	if _, _, _, err := splitFusedQKV(wrong, 2); err == nil {
		t.Errorf("expected an error for a c_attn matrix of the wrong width")
	}
}

func TestSplitFusedBias(t *testing.T) {
	fused, err := NewMatrixFromData(1, 6, []float32{1, 2, 10, 20, 100, 200})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	bq, bk, bv, err := splitFusedBias(fused, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if bq.data[0] != 1 || bq.data[1] != 2 {
		t.Errorf("bq: expected [1 2], got %v", bq.data)
	}
	if bk.data[0] != 10 || bk.data[1] != 20 {
		t.Errorf("bk: expected [10 20], got %v", bk.data)
	}
	if bv.data[0] != 100 || bv.data[1] != 200 {
		t.Errorf("bv: expected [100 200], got %v", bv.data)
	}
}
