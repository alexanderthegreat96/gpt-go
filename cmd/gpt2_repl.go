package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/alexanderthegreat96/gpt-go/bpe"
	"github.com/alexanderthegreat96/gpt-go/gpt"
	"github.com/spf13/cobra"
)

var (
	gpt2ReplDir   string
	gpt2ReplHeads int
	gpt2ReplInt8  bool
)

// gpt2-repl exists because loading 1.5GB of weights for every single prompt is
// the main friction left. load once, prompt as many times as you like.
//
// output streams token by token rather than appearing all at once, which at
// ~21 tok/s is the difference between feeling alive and feeling broken
var gpt2ReplCmd = &cobra.Command{
	Use:   "gpt2-repl",
	Short: "Interactive prompt loop against a pretrained GPT-2, loaded once",
	Run: func(cmd *cobra.Command, args []string) {
		weights := gpt2ReplDir + "/model.safetensors"

		tokenizer, err := bpe.LoadGPT2Tokenizer(gpt2ReplDir+"/vocab.json", gpt2ReplDir+"/merges.txt")
		if err != nil {
			fmt.Printf("Failed to load the tokenizer: %v\n", err)
			return
		}

		fmt.Printf("Loading %s ...\n", weights)
		loadStart := time.Now()
		model, err := gpt.LoadGPT2(weights, gpt.GPT2Config{NumHeads: gpt2ReplHeads, Quantize: gpt2ReplInt8})
		if err != nil {
			fmt.Printf("Failed to load the weights: %v\n", err)
			return
		}

		vocabSize, dModel, numLayers, numHeads, maxSeqLen := model.Describe()

		gpt.ReportQuantization(model)
		if tokenizer.VocabSize() != vocabSize {
			fmt.Printf("The tokenizer has %d tokens but the model has %d embedding rows -> different models.\n",
				tokenizer.VocabSize(), vocabSize)
			return
		}

		fmt.Printf("%s params, %d layers, %d wide, %d heads, %d context. Loaded in %.1fs on %s.\n",
			gpt.FormatParamCount(model.CountParameters()), numLayers, dModel, numHeads, maxSeqLen,
			time.Since(loadStart).Seconds(), gpt.GetActiveBackend())
		fmt.Println()
		fmt.Println("Type a prompt and press enter. /help for the settings, /exit to quit.")
		fmt.Println()

		session := &replSession{
			tokenizer:   tokenizer,
			model:       model,
			maxSeqLen:   maxSeqLen,
			newTokens:   60,
			temperature: 0.8,
			topK:        40,
			repPenalty:  1.1,
			slide:       true,
			stopAtEOS:   true,
		}

		reader := bufio.NewReader(os.Stdin)
		for {
			fmt.Print("> ")

			line, err := reader.ReadString('\n')
			if err != nil {
				// EOF, which is ctrl+d or a piped script running out
				fmt.Println()
				return
			}

			line = strings.TrimRight(line, "\r\n")
			// a byte order mark arrives when input is piped from something
			// that likes to add one, and it would otherwise become a token
			line = strings.TrimPrefix(line, "\uFEFF")
			if strings.TrimSpace(line) == "" {
				continue
			}

			if strings.HasPrefix(line, "/") {
				if session.command(line) {
					return
				}
				continue
			}

			session.generate(line)
		}
	},
}

// replSession holds the settings the slash commands fiddle with
type replSession struct {
	tokenizer *bpe.GPT2Tokenizer
	model     *gpt.GPTModel
	maxSeqLen int

	newTokens   int
	temperature float64
	topK        int
	repPenalty  float64
	slide       bool
	stopAtEOS   bool
}

// command handles a /slash line. returns true when it is time to quit
func (s *replSession) command(line string) bool {
	fields := strings.Fields(line)
	name := fields[0]

	// a helper for the "/thing <number>" shape, which most of these are
	number := func() (float64, bool) {
		if len(fields) < 2 {
			fmt.Printf("  %s needs a value\n", name)
			return 0, false
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			fmt.Printf("  %q is not a number\n", fields[1])
			return 0, false
		}
		return value, true
	}

	switch name {
	case "/exit", "/quit", "/q":
		return true

	case "/help", "/h", "/?":
		fmt.Println("  /tokens N      how many tokens to generate      (now", s.newTokens, ")")
		fmt.Println("  /temp F        sampling temperature, 0 = greedy (now", s.temperature, ")")
		fmt.Println("  /topk N        top-k pool, 0 = everything       (now", s.topK, ")")
		fmt.Println("  /reppen F      repetition penalty, 1 = none     (now", s.repPenalty, ")")
		fmt.Println("  /slide         toggle sliding past the context  (now", s.slide, ")")
		fmt.Println("  /eos           toggle stopping on end-of-text   (now", s.stopAtEOS, ")")
		fmt.Println("  /settings      print all of the above")
		fmt.Println("  /tokenize TEXT show how TEXT splits into tokens")
		fmt.Println("  /exit          quit")

	case "/settings":
		fmt.Printf("  tokens %d, temp %.2f, top-k %d, rep-penalty %.2f, slide %v, stop-on-eos %v\n",
			s.newTokens, s.temperature, s.topK, s.repPenalty, s.slide, s.stopAtEOS)

	case "/tokens":
		if value, ok := number(); ok {
			if value < 1 {
				fmt.Println("  that needs to be at least 1")
				break
			}
			s.newTokens = int(value)
			fmt.Printf("  tokens = %d\n", s.newTokens)
		}

	case "/temp":
		if value, ok := number(); ok {
			s.temperature = value
			fmt.Printf("  temp = %.2f%s\n", s.temperature, map[bool]string{true: " (greedy)"}[value <= 0])
		}

	case "/topk":
		if value, ok := number(); ok {
			s.topK = int(value)
			fmt.Printf("  top-k = %d\n", s.topK)
		}

	case "/reppen":
		if value, ok := number(); ok {
			s.repPenalty = value
			fmt.Printf("  rep-penalty = %.2f\n", s.repPenalty)
		}

	case "/slide":
		s.slide = !s.slide
		fmt.Printf("  slide = %v\n", s.slide)

	case "/eos":
		s.stopAtEOS = !s.stopAtEOS
		fmt.Printf("  stop-on-eos = %v\n", s.stopAtEOS)

	case "/tokenize":
		text := strings.TrimSpace(strings.TrimPrefix(line, "/tokenize"))
		if text == "" {
			fmt.Println("  give it some text")
			break
		}
		ids := s.tokenizer.Encode(text)
		fmt.Printf("  %d tokens %v\n", len(ids), ids)
		for _, id := range ids {
			fmt.Printf("    %5d -> %q\n", id, string(s.tokenizer.GetTokenAtId(id)))
		}

	default:
		fmt.Printf("  unknown command %q, try /help\n", name)
	}

	return false
}

// generate runs one prompt and streams the result
func (s *replSession) generate(prompt string) {
	promptIDs := s.tokenizer.Encode(prompt)
	if len(promptIDs) == 0 {
		fmt.Println("  that encoded to nothing")
		return
	}
	if len(promptIDs) >= s.maxSeqLen {
		fmt.Printf("  that prompt is %d tokens, the context is only %d\n", len(promptIDs), s.maxSeqLen)
		return
	}

	// a single token can be half of a utf-8 sequence, because the tokenizer
	// works on bytes. so we accumulate bytes and only print whole characters
	// -> printing per token would spray replacement characters through any
	// accented or non-latin output
	var pending []byte

	flush := func(force bool) {
		for len(pending) > 0 {
			r, size := utf8.DecodeRune(pending)
			if r == utf8.RuneError && size <= 1 && !force {
				// an incomplete sequence: wait for the next token's bytes
				return
			}
			fmt.Print(string(pending[:size]))
			pending = pending[size:]
		}
	}

	cfg := gpt.GenerateConfig{
		Temperature:       float32(s.temperature),
		TopK:              s.topK,
		RepetitionPenalty: float32(s.repPenalty),
		SlideContext:      s.slide,
		OnToken: func(id int) {
			if s.stopAtEOS && id == bpe.GPT2EOSTokenID {
				return
			}
			pending = append(pending, s.tokenizer.GetTokenAtId(id)...)
			flush(false)
		},
	}
	if s.stopAtEOS {
		cfg.StopTokenIDs = []int{bpe.GPT2EOSTokenID}
	}

	// echo the prompt so the output reads as one continuous piece of text
	fmt.Print(prompt)

	start := time.Now()
	outputIDs, err := s.model.Generate(promptIDs, s.newTokens, cfg)
	if err != nil {
		fmt.Printf("\n  generation failed: %v\n", err)
		return
	}

	// anything left is a genuinely invalid byte sequence, so print it rather
	// than swallow it
	flush(true)

	elapsed := time.Since(start).Seconds()
	produced := len(outputIDs) - len(promptIDs)

	fmt.Println()
	if produced > 0 && elapsed > 0 {
		note := ""
		if outputIDs[len(outputIDs)-1] == bpe.GPT2EOSTokenID {
			note = ", stopped on end-of-text"
		}
		fmt.Printf("  [%d tokens, %.1fs, %.1f tok/s%s]\n", produced, elapsed, float64(produced)/elapsed, note)
	}
	fmt.Println()
}

func init() {
	gpt2ReplCmd.Flags().StringVarP(&gpt2ReplDir, "dir", "d", "models/gpt2", "Directory holding model.safetensors, vocab.json and merges.txt")
	gpt2ReplCmd.Flags().IntVar(&gpt2ReplHeads, "heads", 0, "Attention head count (0 = infer from the hidden dim)")
	gpt2ReplCmd.Flags().BoolVar(&gpt2ReplInt8, "int8", false, "Quantize the matmul weights to int8 -> about 4x less memory, a little slower, ~1% error")

	rootCmd.AddCommand(gpt2ReplCmd)
}
