package bpe

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// GPT-2 byte level BPE
//
// this is a different animal from the Vocabulary in tokenizer.go. that one we
// train ourselves on a corpus and it merges raw byte ids. this one has to
// reproduce OpenAI's tokenizer exactly, because the 50257 embedding rows in a
// pretrained checkpoint are indexed by ITS token ids. one token out of place
// and the model is answering a different prompt than the one you typed.
//
// three things make it awkward:
//
// 1. it is "byte level" -> arbitrary bytes have to survive a round trip, but
//    BPE wants to work on text. so every byte is first mapped to a printable
//    codepoint through a fixed 256 entry table, merges happen on those, and
//    decoding maps back. see buildByteRuneTables
//
// 2. the merges are RANKED. merges.txt is ordered, and the line number is the
//    priority -> at each step you merge the lowest ranked adjacent pair in the
//    whole word, not the first one you find
//
// 3. text is split into pieces BEFORE any merging, by a regex that Go cannot
//    compile. see splitPieces for the gory details
//
// correctness here is checked against fixtures generated from the real
// tokenizer, in gpt2_test.go. that is the only way to know -> a tokenizer that
// is subtly wrong still emits perfectly valid ids

// GPT2EOSTokenID is the end of text token. gpt-2 has exactly one special
// token and this is it, used as BOS, EOS and padding all at once
const GPT2EOSTokenID = 50256

// GPT2EOSToken is the same thing in its written form, which is how it appears
// in vocab.json and in any text that wants to embed it
const GPT2EOSToken = "<|endoftext|>"

type GPT2Tokenizer struct {
	// token text (in the mapped alphabet, NOT raw bytes) -> id, and back
	encoder map[string]int
	decoder map[int]string

	// merge pair -> rank. lower rank wins, and rank is the line number in
	// merges.txt
	ranks map[[2]string]int

	// the fixed byte <-> codepoint mapping
	byteToRune [256]rune
	runeToByte map[rune]byte

	// merging the same piece over and over is the bulk of the work in any
	// real text, so we remember what each one turned into. the reference
	// python implementation does the same
	cache map[string][]int
}

// LoadGPT2Tokenizer reads the vocab.json and merges.txt that ship alongside a
// pretrained checkpoint. all four gpt-2 sizes share one tokenizer, so these
// files are interchangeable between them
func LoadGPT2Tokenizer(vocabPath, mergesPath string) (*GPT2Tokenizer, error) {
	vocabBytes, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("gpt2 tokenizer: reading vocab: %v", err)
	}

	var encoder map[string]int
	if err := json.Unmarshal(vocabBytes, &encoder); err != nil {
		return nil, fmt.Errorf("gpt2 tokenizer: vocab.json is not a string->int map: %v", err)
	}
	if len(encoder) == 0 {
		return nil, fmt.Errorf("gpt2 tokenizer: vocab.json is empty")
	}

	decoder := make(map[int]string, len(encoder))
	for token, id := range encoder {
		decoder[id] = token
	}

	ranks, err := loadMerges(mergesPath)
	if err != nil {
		return nil, err
	}

	byteToRune, runeToByte := buildByteRuneTables()

	return &GPT2Tokenizer{
		encoder:    encoder,
		decoder:    decoder,
		ranks:      ranks,
		byteToRune: byteToRune,
		runeToByte: runeToByte,
		cache:      make(map[string][]int),
	}, nil
}

// loadMerges reads merges.txt into pair -> rank.
//
// the file starts with a "#version:" comment line, then one merge per line as
// two space separated symbols. the ORDER is the whole point: line 0 is the
// highest priority merge
func loadMerges(path string) (map[[2]string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gpt2 tokenizer: reading merges: %v", err)
	}
	defer file.Close()

	ranks := make(map[[2]string]int, 50000)
	scanner := bufio.NewScanner(file)

	// some merge lines are long-ish and the default 64KB buffer is plenty,
	// but be explicit rather than hitting a surprise on an unusual vocab
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	rank := 0
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r\n")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			return nil, fmt.Errorf("gpt2 tokenizer: merge line %d is %q, expected two symbols", rank, line)
		}

		ranks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("gpt2 tokenizer: scanning merges: %v", err)
	}
	if rank == 0 {
		return nil, fmt.Errorf("gpt2 tokenizer: %s contained no merges", path)
	}

	return ranks, nil
}

// buildByteRuneTables constructs GPT-2's fixed byte <-> codepoint mapping.
//
// the problem it solves: BPE is defined over text, but we need to tokenize
// arbitrary bytes without ever producing an unrepresentable or ambiguous
// character. so the 188 bytes that are already printable keep their own
// codepoint, and the other 68 (control characters, space, and a couple of
// holes in latin-1) are moved up to U+0100 and beyond.
//
// that is why a space shows up as the Ġ you see all over merges.txt and
// vocab.json -> byte 0x20 maps to U+0120
func buildByteRuneTables() ([256]rune, map[rune]byte) {
	var byteToRune [256]rune

	// the bytes that keep their own codepoint: printable ascii, then the two
	// printable stretches of latin-1 (0xad is a soft hyphen and is excluded)
	keepsOwn := make([]bool, 256)
	for _, r := range [][2]int{{'!', '~'}, {0xa1, 0xac}, {0xae, 0xff}} {
		for b := r[0]; b <= r[1]; b++ {
			keepsOwn[b] = true
		}
	}

	for b := 0; b < 256; b++ {
		if keepsOwn[b] {
			byteToRune[b] = rune(b)
		}
	}

	// everything else gets pushed into the private-ish range above 255, in
	// ascending byte order
	next := 0
	for b := 0; b < 256; b++ {
		if !keepsOwn[b] {
			byteToRune[b] = rune(256 + next)
			next++
		}
	}

	runeToByte := make(map[rune]byte, 256)
	for b := 0; b < 256; b++ {
		runeToByte[byteToRune[b]] = byte(b)
	}

	return byteToRune, runeToByte
}

// Encode turns text into gpt-2 token ids
func (t *GPT2Tokenizer) Encode(text string) []int {
	var ids []int

	// the special token is matched before the pre-tokenizer gets a look in.
	// without this, the literal text "<|endoftext|>" comes back as the seven
	// ordinary tokens it is made of rather than the single id 50256.
	//
	// worth knowing that openai's original encoder.py did NOT do this -> it
	// treated the text as text. huggingface added the special casing and that
	// is now the de facto behaviour, so it is what we match
	for {
		index := strings.Index(text, GPT2EOSToken)
		if index == -1 {
			break
		}

		ids = append(ids, t.encodeOrdinary(text[:index])...)
		ids = append(ids, GPT2EOSTokenID)
		text = text[index+len(GPT2EOSToken):]
	}

	return append(ids, t.encodeOrdinary(text)...)
}

// encodeOrdinary handles a span of text known to contain no special tokens
func (t *GPT2Tokenizer) encodeOrdinary(text string) []int {
	var ids []int

	for _, piece := range splitPieces(text) {
		ids = append(ids, t.encodePiece(piece)...)
	}

	return ids
}

// encodePiece runs one pre-tokenized piece through the byte mapping and the
// merge table
func (t *GPT2Tokenizer) encodePiece(piece string) []int {
	if cached, ok := t.cache[piece]; ok {
		return cached
	}

	// every byte of the piece becomes one mapped rune. note we walk BYTES
	// here, not runes -> that is what makes this byte level, and what lets
	// an emoji survive as its four constituent bytes
	symbols := make([]string, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		symbols = append(symbols, string(t.byteToRune[piece[i]]))
	}

	symbols = t.applyMerges(symbols)

	ids := make([]int, 0, len(symbols))
	for _, symbol := range symbols {
		id, ok := t.encoder[symbol]
		if !ok {
			// should be unreachable: every single mapped character is in the
			// vocab, so any merge result is too. if it ever fires, the vocab
			// and merges files are from different tokenizers
			continue
		}
		ids = append(ids, id)
	}

	t.cache[piece] = ids
	return ids
}

// applyMerges is the BPE loop itself.
//
// repeatedly find the adjacent pair with the LOWEST rank anywhere in the word
// and merge every occurrence of it, until no adjacent pair has a rank at all.
// merging the lowest ranked pair rather than the first one is the detail that
// matters -> picking the first match produces plausible but wrong tokens
func (t *GPT2Tokenizer) applyMerges(symbols []string) []string {
	if len(symbols) < 2 {
		return symbols
	}

	for {
		bestRank := -1
		var bestPair [2]string

		for i := 0; i < len(symbols)-1; i++ {
			pair := [2]string{symbols[i], symbols[i+1]}
			rank, ok := t.ranks[pair]
			if !ok {
				continue
			}
			if bestRank == -1 || rank < bestRank {
				bestRank = rank
				bestPair = pair
			}
		}

		if bestRank == -1 {
			return symbols
		}

		merged := make([]string, 0, len(symbols))
		for i := 0; i < len(symbols); {
			if i < len(symbols)-1 && symbols[i] == bestPair[0] && symbols[i+1] == bestPair[1] {
				merged = append(merged, bestPair[0]+bestPair[1])
				i += 2
				continue
			}
			merged = append(merged, symbols[i])
			i++
		}

		symbols = merged
		if len(symbols) < 2 {
			return symbols
		}
	}
}

// Decode turns token ids back into text
func (t *GPT2Tokenizer) Decode(ids []int) string {
	var mapped strings.Builder
	for _, id := range ids {
		token, ok := t.decoder[id]
		if !ok {
			continue
		}
		mapped.WriteString(token)
	}

	// and back from codepoints to the original bytes
	raw := make([]byte, 0, mapped.Len())
	for _, r := range mapped.String() {
		b, ok := t.runeToByte[r]
		if !ok {
			continue
		}
		raw = append(raw, b)
	}

	return string(raw)
}

// GetTokenAtId returns the raw bytes a single token stands for, which is handy
// for printing a token breakdown the way cmd/run.go does
func (t *GPT2Tokenizer) GetTokenAtId(id int) []byte {
	token, ok := t.decoder[id]
	if !ok {
		return nil
	}

	raw := make([]byte, 0, len(token))
	for _, r := range token {
		if b, ok := t.runeToByte[r]; ok {
			raw = append(raw, b)
		}
	}
	return raw
}

// VocabSize is how many ids exist, which should match the checkpoint's
// embedding rows
func (t *GPT2Tokenizer) VocabSize() int {
	return len(t.encoder)
}

// splitPieces is GPT-2's pre-tokenizer, hand written because Go will not
// compile the original.
//
// the real pattern is:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// and `(?!` is a negative lookahead, which RE2 rejects outright -> Go's regexp
// package returns "invalid or unsupported Perl syntax". so we walk the string
// and apply the same alternation in the same order, first match wins.
//
// the branch worth understanding is the second to last one. `\s+(?!\S)` means
// "a whitespace run NOT followed by a non-space", and because the regex engine
// backtracks, what it actually does to a run of N spaces followed by a word is
// match the first N-1 and leave the last one to be picked up by ` ?\p{L}+`.
// so the final space attaches to the word that follows:
//
//	"hello  world" -> ["hello", " ", " world"]
//
// drop the lookahead and you get ["hello", "  ", "world"] instead, which
// tokenizes to different ids and quietly feeds the model a different prompt
func splitPieces(text string) []string {
	runes := []rune(text)
	var pieces []string

	for i := 0; i < len(runes); {
		// 1. the contraction branches, which come first and so win
		if n := matchContraction(runes, i); n > 0 {
			pieces = append(pieces, string(runes[i:i+n]))
			i += n
			continue
		}

		// 2, 3 and 4: an optional single SPACE (not any whitespace) followed
		// by a run of letters, or digits, or of anything that is neither
		start := i
		offset := i
		if runes[offset] == ' ' && offset+1 < len(runes) {
			offset++
		}

		if n := runLength(runes, offset, unicode.IsLetter); n > 0 {
			i = offset + n
			pieces = append(pieces, string(runes[start:i]))
			continue
		}
		if n := runLength(runes, offset, unicode.IsNumber); n > 0 {
			i = offset + n
			pieces = append(pieces, string(runes[start:i]))
			continue
		}
		if n := runLength(runes, offset, isOtherSymbol); n > 0 {
			i = offset + n
			pieces = append(pieces, string(runes[start:i]))
			continue
		}

		// 5. a whitespace run, which is all that can be left
		end := i
		for end < len(runes) && unicode.IsSpace(runes[end]) {
			end++
		}

		if end == i {
			// not whitespace and matched nothing above. unreachable for valid
			// utf-8, but emit the character rather than spinning forever
			pieces = append(pieces, string(runes[i]))
			i++
			continue
		}

		switch {
		case end == len(runes):
			// nothing follows, so the lookahead is satisfied by the whole run
			pieces = append(pieces, string(runes[i:end]))
			i = end
		case end-1 > i:
			// a run of 2 or more before a non-space -> hold back the last one
			// for the next piece to claim
			pieces = append(pieces, string(runes[i:end-1]))
			i = end - 1
		default:
			// exactly one whitespace character before a non-space. the
			// lookahead branch cannot match (it needs at least one character
			// left over) so the plain `\s+` branch takes it. only reachable
			// for tabs and newlines -> a single SPACE before a word was
			// already swallowed by the ` ?\p{L}+` branch above
			pieces = append(pieces, string(runes[i:end]))
			i = end
		}
	}

	return pieces
}

// gpt2Contractions are matched literally and in this order, exactly as the
// original pattern lists them. lowercase only, which is why "DON'T" tokenizes
// differently from "don't" -> a quirk of the real tokenizer that we have to
// reproduce rather than fix
var gpt2Contractions = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// matchContraction returns how many runes the matching contraction covers, or
// 0 for no match
func matchContraction(runes []rune, i int) int {
	if runes[i] != '\'' {
		return 0
	}

	rest := runes[i:]
	for _, contraction := range gpt2Contractions {
		candidate := []rune(contraction)
		if len(candidate) > len(rest) {
			continue
		}
		matched := true
		for j, r := range candidate {
			if rest[j] != r {
				matched = false
				break
			}
		}
		if matched {
			return len(candidate)
		}
	}

	return 0
}

// runLength counts how many consecutive runes from i satisfy the predicate
func runLength(runes []rune, i int, predicate func(rune) bool) int {
	n := 0
	for i+n < len(runes) && predicate(runes[i+n]) {
		n++
	}
	return n
}

// isOtherSymbol is the [^\s\p{L}\p{N}] class -> anything that is not
// whitespace, a letter or a number
func isOtherSymbol(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}
