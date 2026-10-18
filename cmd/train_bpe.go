package cmd

import (
	"fmt"
	"time"

	"github.com/alexanderthegreat96/gpt-go/bpe"
	"github.com/spf13/cobra"
)

var (
	bpeCapacity   int
	bpeMaxVocab   int
	bpeInputFile  string
	bpeOutputFile string
)

var trainBpeCmd = &cobra.Command{
	Use:   "train-bpe",
	Short: "Train the custom BPE vocabulary from a text corpus",
	Run: func(cmd *cobra.Command, args []string) {
		now := time.Now()

		fmt.Println("Initializing BPE Training...")
		fmt.Printf("Training using: %s\n", bpeInputFile)

		vocabulary := bpe.NewVocabulary(bpeCapacity)

		specialTokens := []string{
			"[PAD]",
			"[UNK]",
			"<|endoftext|>",
		}

		for _, token := range specialTokens {
			vocabulary.AddSpecialTokenFromString(token)
		}

		trainData, err := bpe.LoadTrainingData(bpeInputFile)
		if err != nil {
			fmt.Printf("Failed to load training data from %s: %v\n", bpeInputFile, err)
			return
		}

		vocabulary.TrainFromBytes(trainData, bpeMaxVocab)

		if err := bpe.SaveTokenizer(vocabulary, bpeOutputFile); err != nil {
			fmt.Printf("Failed to save tokenizer: %v\n", err)
			return
		}

		took := time.Since(now).Abs().Seconds()
		fmt.Printf("Training Finished. Saved to %s. Took: %.2f seconds\n", bpeOutputFile, took)
	},
}

func init() {
	trainBpeCmd.Flags().IntVar(&bpeCapacity, "capacity", 300, "Vocabulary capacity")
	trainBpeCmd.Flags().IntVar(&bpeMaxVocab, "max-vocab", 2000, "Maximum vocabulary size during training")
	trainBpeCmd.Flags().StringVarP(&bpeInputFile, "input", "i", "models/tiny_wiki/train.txt", "Path to training text corpus file")
	trainBpeCmd.Flags().StringVarP(&bpeOutputFile, "output", "o", "tokenizer_model_data.json", "Path to save tokenizer JSON")

	rootCmd.AddCommand(trainBpeCmd)
}
