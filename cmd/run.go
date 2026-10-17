package cmd

import (
	"fmt"
	"strings"

	"github.com/alexanderthegreat96/gpt-go/bpe"
	"github.com/alexanderthegreat96/gpt-go/gpt"
	"github.com/spf13/cobra"
)

var (
	runPrompt        string
	runTokenizerFile string
	runModelFile     string
	runMaxNewTokens  int
	runTemperature   float64
	runTopK          int
	runRepPenalty    float64
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run autoregressive text generation using the trained GPT model",
	Run: func(cmd *cobra.Command, args []string) {
		vocab, err := bpe.LoadTokenizer(runTokenizerFile)
		if err != nil {
			fmt.Printf("Tokenizer file not found at %s (%v). Training default tokenizer on sample corpus...\n", runTokenizerFile, err)
			vocab = bpe.NewVocabulary(300)
			vocab.TrainFromString("Hello world, this is a test of the mini gpt model built entirely from scratch in pure Go.", 300)
		}

		encoded := vocab.Encode(runPrompt)

		fmt.Printf("Prompt: %s\n", runPrompt)
		fmt.Printf("Using Tokenizer File: %s\n", runTokenizerFile)
		fmt.Printf("Using Model File: %s\n", runModelFile)
		fmt.Printf("Encoded Prompt Tokens: %v\n", encoded)

		model, err := gpt.LoadModel(runModelFile)
		if err != nil {
			fmt.Printf("Model file not found at %s (%v). Initializing fallback random model for demo...\n", runModelFile, err)
			vocabSize := len(vocab.GetData())
			model, err = gpt.NewGPTModel(vocabSize, 16, 2, 4, 512, 0.1)
			if err != nil {
				fmt.Printf("Failed to create fallback GPT model: %v\n", err)
				return
			}
		}

		cfg := gpt.GenerateConfig{
			Temperature:       float32(runTemperature),
			TopK:              runTopK,
			RepetitionPenalty: float32(runRepPenalty),
		}

		fmt.Printf("Generating %d new tokens (temp=%.2f, top-k=%d, rep-penalty=%.2f)...\n",
			runMaxNewTokens, cfg.Temperature, cfg.TopK, cfg.RepetitionPenalty)

		generatedTokens, err := model.Generate(encoded, runMaxNewTokens, cfg)
		if err != nil {
			fmt.Printf("Generation failed: %v\n", err)
			return
		}

		generatedText := vocab.Decode(generatedTokens)

		fmt.Printf("Generated Sequence IDs : %v\n", generatedTokens)
		fmt.Printf("Final Decoded Output   :\n\n%s\n", strings.ReplaceAll(generatedText, "\r", ""))

		fmt.Println("\nToken Breakdown:")
		for _, id := range generatedTokens {
			tokenBytes := vocab.GetTokenAtId(id)
			fmt.Printf("  ID %3d -> %q\n", id, tokenBytes)
		}
	},
}

func init() {
	runCmd.Flags().StringVarP(&runPrompt, "prompt", "p", "Hello world", "Prompt text for generation")
	runCmd.Flags().StringVarP(&runTokenizerFile, "tokenizer", "t", "models/tiny_wiki/tokenizer_model_data.json", "Path to tokenizer JSON file")
	runCmd.Flags().StringVarP(&runModelFile, "model", "m", "models/tiny_wiki/gpt_model.json", "Path to trained GPT model JSON file")
	runCmd.Flags().IntVarP(&runMaxNewTokens, "tokens", "n", 8, "Maximum new tokens to generate")
	runCmd.Flags().Float64Var(&runTemperature, "temperature", 1.0, "Sampling temperature (0 = greedy argmax)")
	runCmd.Flags().IntVarP(&runTopK, "top-k", "k", 40, "Top-k sampling pool size (0 = full distribution)")
	runCmd.Flags().Float64Var(&runRepPenalty, "rep-penalty", 1.0, "Repetition penalty (1.0 = none, >1 discourages repeats)")

	rootCmd.AddCommand(runCmd)
}
