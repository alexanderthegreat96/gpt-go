package gpt

import "testing"

func TestGPTModelGenerate(t *testing.T) {
	vocabSize := 50
	dModel := 8
	numLayers := 1
	numHeads := 2
	scale := float32(0.1)

	model, err := NewGPTModel(vocabSize, dModel, numLayers, numHeads, 128, scale)
	if err != nil {
		t.Fatalf("failed to create GPT model for generation test: %v", err)
	}

	promptTokenIDs := []int{10, 20, 30}
	maxNewTokens := 5

	generated, err := model.Generate(promptTokenIDs, maxNewTokens, DefaultGenerateConfig())
	if err != nil {
		t.Fatalf("unexpected error during text generation: %v", err)
	}

	expectedLen := len(promptTokenIDs) + maxNewTokens
	if len(generated) != expectedLen {
		t.Errorf("expected generated sequence length %d, got %d", expectedLen, len(generated))
	}

	for i, originalToken := range promptTokenIDs {
		if generated[i] != originalToken {
			t.Errorf("prompt token at index %d was altered: expected %d, got %d", i, originalToken, generated[i])
		}
	}
}
