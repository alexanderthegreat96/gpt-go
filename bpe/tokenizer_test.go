package bpe

import (
	"testing"
)

func TestNewVocabulary(t *testing.T) {
	vocab := NewVocabulary(256)

	if len(vocab.GetData()) != 256 {
		t.Errorf("expected 256 initial vocabulary entries, got %d", len(vocab.GetData()))
	}

	tokenBytes := vocab.GetTokenAtId(65)
	if tokenBytes == nil || string(tokenBytes) != "A" {
		t.Errorf("expected token ID 65 to map to 'A', got %v", tokenBytes)
	}
}

func TestSpecialTokens(t *testing.T) {
	vocab := NewVocabulary(256)
	initialLen := len(vocab.GetData())

	vocab.AddSpecialTokenFromBytes([]byte("<|endoftext|>"))
	if len(vocab.GetData()) != initialLen+1 {
		t.Errorf("expected vocabulary size to increase by 1")
	}

	vocab.AddSpecialTokenFromString("<|pad|>")
	if len(vocab.GetData()) != initialLen+2 {
		t.Errorf("expected vocabulary size to increase by 2")
	}

	retrieved := vocab.GetTokenAtId(initialLen)
	if string(retrieved) != "<|endoftext|>" {
		t.Errorf("expected '<|endoftext|>', got %s", string(retrieved))
	}
}

func TestTrainAndEncodeDecode(t *testing.T) {
	vocab := NewVocabulary(256)

	corpus := "banana bandanna banana bandanna banana"
	maxVocabSize := 260

	vocab.TrainFromString(corpus, maxVocabSize)

	if len(vocab.GetData()) > maxVocabSize {
		t.Errorf("vocabulary size exceeded max limit: got %d, max %d", len(vocab.GetData()), maxVocabSize)
	}

	testStr := "banana"
	encoded := vocab.Encode(testStr)
	if len(encoded) == 0 {
		t.Fatal("encoding returned empty slice")
	}

	decoded := vocab.Decode(encoded)
	if decoded != testStr {
		t.Errorf("round-trip decode failed: expected '%s', got '%s'", testStr, decoded)
	}
}

func TestFormatTokens(t *testing.T) {
	vocab := NewVocabulary(256)

	tokens := vocab.Encode("cat")
	formatted := vocab.FormatTokens(tokens)

	expected := "c_a_t"
	if formatted != expected {
		t.Errorf("expected formatted tokens '%s', got '%s'", expected, formatted)
	}
}
