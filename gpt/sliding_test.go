package gpt

import "testing"

// a deliberately tiny context, so generation runs past the end of it within a
// handful of tokens rather than needing a thousand
const slidingTestContext = 8

func newSlidingTestModel(t *testing.T) *GPTModel {
	t.Helper()

	model, err := NewGPTModel(40, 8, 1, 2, slidingTestContext, 0.1)
	if err != nil {
		t.Fatalf("failed to build the test model: %v", err)
	}
	return model
}

func TestGenerateStopsAtTheContextLimitByDefault(t *testing.T) {
	// the old behaviour, and still the default -> SlideContext is false on a
	// zero value config, so nothing that already calls Generate changes
	model := newSlidingTestModel(t)

	prompt := []int{1, 2, 3}
	asked := 40

	generated, err := model.Generate(prompt, asked, DefaultGenerateConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(generated) >= len(prompt)+asked {
		t.Errorf("expected generation to stop short of %d tokens without sliding, got %d",
			len(prompt)+asked, len(generated))
	}
	if len(generated) > slidingTestContext+1 {
		t.Errorf("expected to stop around the %d token context, got %d tokens",
			slidingTestContext, len(generated))
	}
}

func TestGenerateSlidesPastTheContextLimit(t *testing.T) {
	model := newSlidingTestModel(t)

	prompt := []int{1, 2, 3}
	asked := 40

	cfg := DefaultGenerateConfig()
	cfg.SlideContext = true

	generated, err := model.Generate(prompt, asked, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// with sliding we should get everything we asked for, several times the
	// context length
	if len(generated) != len(prompt)+asked {
		t.Errorf("expected %d tokens with sliding enabled, got %d", len(prompt)+asked, len(generated))
	}

	// the prompt still has to come back untouched at the front, even though
	// the model itself has long since forgotten it
	for i, token := range prompt {
		if generated[i] != token {
			t.Errorf("prompt token %d was altered: expected %d, got %d", i, token, generated[i])
		}
	}
}

func TestSlideContextRenumbersPositionsFromZero(t *testing.T) {
	// the reason sliding rebuilds the cache instead of trimming it: positional
	// embeddings are absolute and already baked into the cached keys. after a
	// slide the cache must hold exactly the retained tokens, numbered from
	// zero, or two tokens end up sharing a position
	model := newSlidingTestModel(t)

	caches := make([]*KvCache, model.numLayers)
	for i := range caches {
		caches[i] = &KvCache{}
	}

	// fill the window right up
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	if _, err := model.ForwardWithCache(tokens, caches); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if caches[0].key.rows != len(tokens) {
		t.Fatalf("expected a cache of %d rows, got %d", len(tokens), caches[0].key.rows)
	}

	logits, err := model.slideContext(tokens, caches)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// half the window is retained, and the cache now holds exactly that many
	// rows -> which is what makes the next token's position correct
	expectedKeep := slidingTestContext / 2
	if caches[0].key.rows != expectedKeep {
		t.Errorf("expected the cache to hold %d rows after sliding, got %d", expectedKeep, caches[0].key.rows)
	}

	// and the logits returned describe the last retained token, so sampling
	// can carry straight on
	if logits.rows != expectedKeep {
		t.Errorf("expected logits for %d retained tokens, got %d rows", expectedKeep, logits.rows)
	}
	if logits.cols != model.vocabSize {
		t.Errorf("expected %d logit columns, got %d", model.vocabSize, logits.cols)
	}

	// every cache layer has to be rebuilt, not just the first
	for i, cache := range caches {
		if cache.key == nil || cache.key.rows != expectedKeep {
			t.Errorf("cache for layer %d was not rebuilt", i)
		}
	}
}

func TestSlideContextHandlesAShortHistory(t *testing.T) {
	// a slide asked for before there are even half a window of tokens should
	// keep what exists rather than slicing out of bounds
	model := newSlidingTestModel(t)

	caches := make([]*KvCache, model.numLayers)
	for i := range caches {
		caches[i] = &KvCache{}
	}

	tokens := []int{1, 2}
	if _, err := model.slideContext(tokens, caches); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if caches[0].key.rows != len(tokens) {
		t.Errorf("expected all %d tokens to be retained, got %d", len(tokens), caches[0].key.rows)
	}
}

func TestGenerateStopsOnAStopToken(t *testing.T) {
	model := newSlidingTestModel(t)

	// greedy sampling makes this deterministic, then we declare whatever it
	// produces first to be a stop token and check it halts there
	cfg := DefaultGenerateConfig()
	cfg.Temperature = 0

	prompt := []int{1, 2, 3}

	first, err := model.Generate(prompt, 1, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	stopToken := first[len(first)-1]

	cfg.StopTokenIDs = []int{stopToken}
	generated, err := model.Generate(prompt, 5, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// the stop token is kept, since it is part of the output, but nothing
	// comes after it
	if len(generated) != len(prompt)+1 {
		t.Errorf("expected generation to stop after the first token, got %d tokens", len(generated))
	}
	if generated[len(generated)-1] != stopToken {
		t.Errorf("expected the last token to be the stop token %d, got %d", stopToken, generated[len(generated)-1])
	}
}

func TestStopTokensAreOffByDefault(t *testing.T) {
	// the zero value of StopTokenIDs is nil, which has to mean "never stop".
	// an int field would have made token 0 a stop token for every model we
	// train ourselves, which is a real id in our byte level vocabularies
	cfg := DefaultGenerateConfig()

	if cfg.isStopToken(0) {
		t.Errorf("token 0 should not be a stop token on a default config")
	}
	for _, id := range []int{1, 50256, -1} {
		if cfg.isStopToken(id) {
			t.Errorf("token %d should not be a stop token on a default config", id)
		}
	}
}
