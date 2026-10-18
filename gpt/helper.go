package gpt

import (
	"fmt"
	"strings"
)

// little helper to get the Backend Name
func GetActiveBackend() string {
	return activeBackend.GetBackendName()
}

// ReportQuantization prints what the int8 weights are saving, if anything.
//
// the conversion itself happens during the load, via GPT2Config.Quantize ->
// doing it afterwards leaves the steady state smaller but makes PEAK memory
// worse, because for a moment the float32 weights and their int8 copies are
// both live. measured on gpt2-medium: 1863MB peak as float32, 2169MB when
// quantizing after the load, 644MB when quantizing during it
func ReportQuantization(model *GPTModel) {
	saved := model.QuantizedBytesSaved()
	if saved == 0 {
		return
	}
	fmt.Printf("Quantized : int8 matmul weights, %.0f MB less than float32\n", float64(saved)/(1024*1024))
}

// IsShapeDefiningTensor picks out the handful of tensors that tell you what
// you are looking at. two kinds of file land here, so we cover both:
// a checkpoint (embeddings plus block 0 as a sample) and a reference dump
// (where embed, ln_f and logits carry the shape)
func IsShapeDefiningTensor(name string) bool {
	trimmed := strings.TrimPrefix(name, "transformer.")

	switch {
	case trimmed == "wte.weight", trimmed == "wpe.weight":
		return true
	case trimmed == "ln_f.weight", trimmed == "ln_f.bias", trimmed == "lm_head.weight":
		return true
	case trimmed == "embed", trimmed == "ln_f", trimmed == "logits":
		return true
	case strings.HasPrefix(trimmed, "h.0."):
		return true
	default:
		return false
	}
}

// FormatParamCount prints 124439808 as "124.44M", because nobody can read the
// raw number at a glance
func FormatParamCount(count int) string {
	switch {
	case count >= 1_000_000_000:
		return fmt.Sprintf("%.2fB (%d)", float64(count)/1e9, count)
	case count >= 1_000_000:
		return fmt.Sprintf("%.2fM (%d)", float64(count)/1e6, count)
	case count >= 1_000:
		return fmt.Sprintf("%.2fK (%d)", float64(count)/1e3, count)
	default:
		return fmt.Sprintf("%d", count)
	}
}
