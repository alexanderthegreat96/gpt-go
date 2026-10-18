package gpt

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Safetensors Reader
// this is the format HuggingFace ships pretrained weights in and thankfully
// it's about as simple as a binary format gets:
//
// 1. the first 8 bytes are a little-endian uint64 telling us the header length
// 2. then that many bytes of JSON, mapping tensor name -> dtype, shape and a
//    [start, end] byte range
// 3. everything after the header is raw tensor bytes, tightly packed, row-major
//
// the one detail that will bite you if you skim the spec: those byte ranges are
// relative to the START OF THE DATA SECTION, not the start of the file. so the
// data begins at 8 + headerLen and every offset gets added to that
//
// why bother instead of reusing our own json persistence? because MatrixDTO
// costs us roughly 11 bytes per parameter. gpt-2 small would be a ~1.4GB json
// file to parse on every single run. this is 4 bytes per param and we slice it
// straight out of the file buffer

// a single tensor's entry in the json header
type stEntry struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets []uint64 `json:"data_offsets"`
}

// how many elements we pull off disk at a time. tensors get read in bounded
// chunks rather than all at once, so a 206MB embedding table does not need a
// 206MB byte buffer alongside the float32 slice it is being decoded into
const tensorChunkElements = 256 * 1024

type SafeTensors struct {
	// we hold the file open and read tensors out of it on demand rather than
	// slurping the whole thing.
	//
	// the obvious implementation is os.ReadFile, and that is what this was. the
	// problem is you then hold the file AND the matrices you build from it at
	// the same time -> gpt2-medium peaked at 3.85GB against a 1.52GB file, and
	// about 1.5GB of that was the file buffer doing nothing but being copied
	// out of. reading per tensor removes it entirely.
	//
	// mmap would be the fancier answer and it is not actually better here:
	// we materialise the qkv split and the lm_head transpose either way, so a
	// mapping of the whole file would be on top of those rather than instead
	// of them. it also needs unsafe and per platform build tags
	file     *os.File
	dataAt   int64 // where the raw tensor bytes start -> 8 + headerLen
	fileSize int64

	entries map[string]stEntry // tensor name -> where to find it

	// safetensors allows an arbitrary string->string metadata blob under the
	// "__metadata__" key. we abuse it in the reference dumper to carry the
	// token ids along with the activations
	Metadata map[string]string
}

// OpenSafeTensors reads a .safetensors file and parses just the header.
// the tensor bytes are kept around as one big buffer and only decoded when
// somebody actually asks for a specific tensor
func OpenSafeTensors(path string) (*SafeTensors, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	// anything that goes wrong from here on has to close the file, since the
	// caller has no handle to close yet
	closeAndFail := func(format string, args ...any) (*SafeTensors, error) {
		file.Close()
		return nil, fmt.Errorf(format, args...)
	}

	info, err := file.Stat()
	if err != nil {
		return closeAndFail("safetensors: cannot stat %s: %v", path, err)
	}
	if info.Size() < 8 {
		return closeAndFail("safetensors: file %s is too short to even hold a header length", path)
	}

	// step 1: how long is the json header
	var lengthBuf [8]byte
	if _, err := file.ReadAt(lengthBuf[:], 0); err != nil {
		return closeAndFail("safetensors: cannot read the header length of %s: %v", path, err)
	}
	headerLen := binary.LittleEndian.Uint64(lengthBuf[:])

	if headerLen > uint64(info.Size())-8 {
		return closeAndFail("safetensors: header claims %d bytes but the file only has %d", headerLen, info.Size()-8)
	}

	// step 2: the header itself. we can't unmarshal straight into
	// map[string]stEntry because "__metadata__" sits in the same object and has
	// a completely different shape, so we go through RawMessage and split them
	headerBytes := make([]byte, headerLen)
	if _, err := file.ReadAt(headerBytes, 8); err != nil {
		return closeAndFail("safetensors: cannot read the header of %s: %v", path, err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return closeAndFail("safetensors: header is not valid json: %v", err)
	}

	st := &SafeTensors{
		file:     file,
		dataAt:   int64(8 + headerLen),
		fileSize: info.Size(),
		entries:  make(map[string]stEntry, len(rawHeader)),
		Metadata: make(map[string]string),
	}

	for name, blob := range rawHeader {
		if name == "__metadata__" {
			// best effort -> a weird metadata blob shouldn't kill the load
			_ = json.Unmarshal(blob, &st.Metadata)
			continue
		}

		var entry stEntry
		if err := json.Unmarshal(blob, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: bad entry for tensor %q: %v", name, err)
		}
		if len(entry.Offsets) != 2 {
			return nil, fmt.Errorf("safetensors: tensor %q has %d offsets, expected 2", name, len(entry.Offsets))
		}
		st.entries[name] = entry
	}

	return st, nil
}

// Close releases the file handle. Matrix calls after this will fail rather
// than read from a closed file
func (st *SafeTensors) Close() {
	if st.file != nil {
		st.file.Close()
		st.file = nil
	}
}

// Names returns every tensor name in the file, sorted so output is stable
func (st *SafeTensors) Names() []string {
	names := make([]string, 0, len(st.entries))
	for name := range st.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Has tells you whether a tensor exists without pulling it out.
// handy for the optional ones -> gpt-2 ties lm_head to wte, so a checkpoint
// may or may not actually carry an lm_head.weight
func (st *SafeTensors) Has(name string) bool {
	_, ok := st.entries[name]
	return ok
}

// Shape returns the stored shape of a tensor as-is
func (st *SafeTensors) Shape(name string) ([]int, error) {
	entry, ok := st.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor named %q", name)
	}
	shape := make([]int, len(entry.Shape))
	copy(shape, entry.Shape)
	return shape, nil
}

// Matrix pulls a tensor out and converts it into one of our matrices.
// we only deal in 2D here, so:
//   - a 1D tensor of N (every bias and layernorm param) becomes 1 x N
//   - a 2D tensor becomes rows x cols directly, no transposing
//
// that second point is the lucky bit of this whole project: huggingface gpt-2
// uses Conv1D rather than nn.Linear, and Conv1D stores its weight as
// (in_features, out_features) with a forward pass of x @ weight + bias.
// that is exactly our input.MatrixMul(weight) convention, so the weights drop
// in without a single transpose. had openai used nn.Linear we would be
// flipping every matrix in the file
func (st *SafeTensors) Matrix(name string) (*Matrix, error) {
	entry, ok := st.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor named %q", name)
	}

	if st.file == nil {
		return nil, fmt.Errorf("safetensors: file was already closed, cannot read %q", name)
	}

	// figure out the 2D shape we are mapping this onto
	var rows, cols int
	switch len(entry.Shape) {
	case 1:
		rows, cols = 1, entry.Shape[0]
	case 2:
		rows, cols = entry.Shape[0], entry.Shape[1]
	default:
		return nil, fmt.Errorf("safetensors: tensor %q has %dD shape %v, we only handle 1D and 2D", name, len(entry.Shape), entry.Shape)
	}

	start := st.dataAt + int64(entry.Offsets[0])
	end := st.dataAt + int64(entry.Offsets[1])
	if start < st.dataAt || end > st.fileSize || start > end {
		return nil, fmt.Errorf("safetensors: tensor %q has byte range [%d, %d] outside the file", name, start, end)
	}

	count := rows * cols
	width, err := dtypeWidth(entry.DType)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %v", name, err)
	}

	if expected := int64(count * width); end-start != expected {
		return nil, fmt.Errorf("safetensors: tensor %q spans %d bytes but shape %v of %s needs %d",
			name, end-start, entry.Shape, entry.DType, expected)
	}

	data := make([]float32, count)

	// read and decode in bounded chunks. the alternative is one buffer the
	// size of the tensor, which for the 206MB embedding table would sit
	// alongside the 206MB of float32 we are decoding into
	chunk := min(count, tensorChunkElements)
	buf := make([]byte, chunk*width)

	for done := 0; done < count; {
		n := min(count-done, chunk)
		raw := buf[:n*width]

		if _, err := st.file.ReadAt(raw, start+int64(done*width)); err != nil {
			return nil, fmt.Errorf("safetensors: reading tensor %q at element %d: %v", name, done, err)
		}
		if err := decodeInto(entry.DType, raw, data[done:done+n]); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %v", name, err)
		}

		done += n
	}

	// built directly rather than through NewMatrixFromData, which copies its
	// input -> we already own this slice, and copying a 206MB tensor just to
	// throw the original away defeats the point of chunking
	return &Matrix{rows: rows, cols: cols, data: data}, nil
}

// dtypeWidth is how many bytes one element of a dtype takes on disk
func dtypeWidth(dtype string) (int, error) {
	switch dtype {
	case "F32":
		return 4, nil
	case "F16", "BF16":
		return 2, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q (we handle F32, F16 and BF16)", dtype)
	}
}

// decodeInto widens raw little-endian tensor bytes into an existing float32
// slice. everything becomes float32 because that is the only thing our Matrix
// speaks -> real int8 inference is a separate quantized matmul path, not a
// decode step here
func decodeInto(dtype string, blob []byte, dst []float32) error {
	width, err := dtypeWidth(dtype)
	if err != nil {
		return err
	}

	if len(blob) != len(dst)*width {
		return fmt.Errorf("expected %d bytes for %d %s values, got %d", len(dst)*width, len(dst), dtype, len(blob))
	}

	switch dtype {
	case "F32":
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
		}
	case "F16":
		for i := range dst {
			dst[i] = halfToFloat32(binary.LittleEndian.Uint16(blob[i*2:]))
		}
	case "BF16":
		// bfloat16 is just float32 with the bottom 16 mantissa bits thrown
		// away, so widening it is a shift and nothing else. this is the whole
		// reason training frameworks like it
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(blob[i*2:])) << 16)
		}
	}

	return nil
}

// halfToFloat32 converts an IEEE 754 half (1 sign, 5 exponent, 10 mantissa)
// into a float32 (1 sign, 8 exponent, 23 mantissa).
// the exponent bias changes from 15 to 127 and the mantissa shifts up by 13
// bits. the only fiddly case is subnormals, where there is no implicit leading
// 1 and we have to normalize by hand
func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch exp {
	case 0:
		if mant == 0 {
			// plus or minus zero
			return math.Float32frombits(sign)
		}
		// subnormal -> shift the mantissa left until we get an implicit
		// leading 1, then pay for each shift out of the exponent
		shifts := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			shifts++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (113-shifts)<<23 | mant<<13)
	case 0x1f:
		// inf or nan -> the exponent goes all-ones in float32 too
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}
