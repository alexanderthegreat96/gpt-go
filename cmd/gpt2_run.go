package cmd

import (
	"fmt"
	"strings"
	"time"

	"github.com/alexanderthegreat96/gpt-go/bpe"
	"github.com/alexanderthegreat96/gpt-go/gpt"
	"github.com/spf13/cobra"
)

var (
	gpt2RunDir       string
	gpt2RunPrompt    string
	gpt2RunTokens    int
	gpt2RunTemp      float64
	gpt2RunTopK      int
	gpt2RunRepPen    float64
	gpt2RunHeads     int
	gpt2RunShowIDs   bool
	gpt2RunIgnoreEOS bool
	gpt2RunSlide     bool
	gpt2RunInt8      bool
)

var gpt2RunCmd = &cobra.Command{
	Use:   "gpt2-run",
	Short: "Generate text with a pretrained GPT-2 checkpoint",
	Long: "Loads a pretrained GPT-2 and generates from a prompt, using GPT-2's own\n" +
		"byte level BPE rather than the tokenizer we train ourselves.\n\n" +
		"Verify the checkpoint first with verify-gpt2 -> a mis-loaded model produces\n" +
		"fluent nonsense that is hard to tell apart from a bad sample.",
	Run: func(cmd *cobra.Command, args []string) {
		vocabPath := gpt2RunDir + "/vocab.json"
		mergesPath := gpt2RunDir + "/merges.txt"
		weightsPath := gpt2RunDir + "/model.safetensors"

		fmt.Printf("Model     : %s\n", weightsPath)
		fmt.Printf("Backend   : %s\n", gpt.GetActiveBackend())

		tokenizer, err := bpe.LoadGPT2Tokenizer(vocabPath, mergesPath)
		if err != nil {
			fmt.Printf("Failed to load the tokenizer: %v\n", err)
			fmt.Println()
			fmt.Println("Download the two files it needs (they are the same for every GPT-2 size):")
			fmt.Printf("  curl -L -o %s \\\n", vocabPath)
			fmt.Println("    https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json")
			fmt.Printf("  curl -L -o %s \\\n", mergesPath)
			fmt.Println("    https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt")
			return
		}

		loadStart := time.Now()
		model, err := gpt.LoadGPT2(weightsPath, gpt.GPT2Config{NumHeads: gpt2RunHeads, Quantize: gpt2RunInt8})
		if err != nil {
			fmt.Printf("Failed to load the weights: %v\n", err)
			return
		}

		vocabSize, dModel, numLayers, numHeads, maxSeqLen := model.Describe()

		gpt.ReportQuantization(model)
		fmt.Printf("Loaded    : %s params, %d layers, %d wide, %d heads, %d context (%.1fs)\n",
			gpt.FormatParamCount(model.CountParameters()), numLayers, dModel, numHeads, maxSeqLen,
			time.Since(loadStart).Seconds())

		// a tokenizer and a checkpoint that disagree on vocabulary size would
		// index the embedding table with ids it does not have, so say so now
		// rather than failing somewhere in the forward pass
		if tokenizer.VocabSize() != vocabSize {
			fmt.Printf("\nThe tokenizer has %d tokens but the model has %d embedding rows.\n",
				tokenizer.VocabSize(), vocabSize)
			fmt.Println("These files are from different models.")
			return
		}

		promptIDs := tokenizer.Encode(gpt2RunPrompt)
		if len(promptIDs) == 0 {
			fmt.Println("\nThe prompt encoded to nothing, give it some text")
			return
		}
		// the prompt itself always has to fit, since it is one forward pass
		if len(promptIDs) >= maxSeqLen {
			fmt.Printf("\nThat prompt is %d tokens and the context is only %d.\n", len(promptIDs), maxSeqLen)
			return
		}
		// the COMBINED length only matters when we are not sliding -> with
		// --slide, running past the window is the entire point
		if !gpt2RunSlide && len(promptIDs)+gpt2RunTokens > maxSeqLen {
			fmt.Printf("\n%d prompt tokens plus %d new ones exceeds the %d token context.\n",
				len(promptIDs), gpt2RunTokens, maxSeqLen)
			fmt.Println("Shorten the prompt, ask for fewer tokens, or pass --slide to keep going.")
			return
		}

		fmt.Printf("Prompt    : %q -> %d tokens %v\n", gpt2RunPrompt, len(promptIDs), promptIDs)

		cfg := gpt.GenerateConfig{
			Temperature:       float32(gpt2RunTemp),
			TopK:              gpt2RunTopK,
			RepetitionPenalty: float32(gpt2RunRepPen),
		}
		if !gpt2RunIgnoreEOS {
			// gpt-2 ends a document with 50256. without this it just starts
			// writing an unrelated one
			cfg.StopTokenIDs = []int{bpe.GPT2EOSTokenID}
		}
		cfg.SlideContext = gpt2RunSlide

		fmt.Printf("Sampling  : %d tokens, temp %.2f, top-k %d, rep-penalty %.2f\n",
			gpt2RunTokens, cfg.Temperature, cfg.TopK, cfg.RepetitionPenalty)
		fmt.Println()

		genStart := time.Now()
		outputIDs, err := model.Generate(promptIDs, gpt2RunTokens, cfg)
		if err != nil {
			fmt.Printf("Generation failed: %v\n", err)
			return
		}
		elapsed := time.Since(genStart).Seconds()

		newTokens := len(outputIDs) - len(promptIDs)
		text := tokenizer.Decode(outputIDs)

		fmt.Println(strings.Repeat("-", 72))
		fmt.Println(text)
		fmt.Println(strings.Repeat("-", 72))
		fmt.Println()

		if newTokens > 0 && elapsed > 0 {
			fmt.Printf("%d new tokens in %.1fs (%.1f tok/s)\n", newTokens, elapsed, float64(newTokens)/elapsed)
		}
		if newTokens > 0 && outputIDs[len(outputIDs)-1] == bpe.GPT2EOSTokenID {
			fmt.Println("Stopped on the end-of-text token.")
		}

		if gpt2RunShowIDs {
			fmt.Println()
			fmt.Println("Token breakdown:")
			for i, id := range outputIDs {
				marker := "  "
				if i == len(promptIDs) {
					marker = "> " // where the model took over
				}
				fmt.Printf("%s%5d -> %q\n", marker, id, string(tokenizer.GetTokenAtId(id)))
			}
		}
	},
}

func init() {
	gpt2RunCmd.Flags().StringVarP(&gpt2RunDir, "dir", "d", "models/gpt2", "Directory holding model.safetensors, vocab.json and merges.txt")
	gpt2RunCmd.Flags().StringVarP(&gpt2RunPrompt, "prompt", "p", "The area of Antarctica is", "Prompt to generate from")
	gpt2RunCmd.Flags().IntVarP(&gpt2RunTokens, "tokens", "n", 40, "How many new tokens to generate")
	gpt2RunCmd.Flags().Float64Var(&gpt2RunTemp, "temperature", 0.8, "Sampling temperature (0 = greedy argmax)")
	gpt2RunCmd.Flags().IntVarP(&gpt2RunTopK, "top-k", "k", 40, "Top-k sampling pool size (0 = full distribution)")
	gpt2RunCmd.Flags().Float64Var(&gpt2RunRepPen, "rep-penalty", 1.0, "Repetition penalty (1.0 = none)")
	gpt2RunCmd.Flags().IntVar(&gpt2RunHeads, "heads", 0, "Attention head count (0 = infer from the hidden dim)")
	gpt2RunCmd.Flags().BoolVar(&gpt2RunShowIDs, "show-tokens", false, "Print the token by token breakdown")
	gpt2RunCmd.Flags().BoolVar(&gpt2RunIgnoreEOS, "ignore-eos", false, "Keep generating past the end-of-text token")
	gpt2RunCmd.Flags().BoolVar(&gpt2RunSlide, "slide", false, "Keep going past the context limit by sliding the window instead of stopping")
	gpt2RunCmd.Flags().BoolVar(&gpt2RunInt8, "int8", false, "Quantize the matmul weights to int8 -> about 4x less memory, a little slower, ~1% error")

	rootCmd.AddCommand(gpt2RunCmd)
}
