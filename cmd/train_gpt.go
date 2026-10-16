package cmd

import (
	"fmt"
	"time"

	"github.com/alexanderthegreat96/gpt-go/bpe"
	"github.com/alexanderthegreat96/gpt-go/gpt"
	"github.com/spf13/cobra"
)

var (
	gptInputFile     string
	gptTokenizerFile string
	gptOutputFile    string
	gptEpochs        int
	gptLR            float64
	gptMaxSeqLen     int
	gptPrintEvery    int
)

var trainGptCmd = &cobra.Command{
	Use:   "train-gpt",
	Short: "Train the GPT model using backpropagation and AdamW",
	Run: func(cmd *cobra.Command, args []string) {
		vocab, err := bpe.LoadTokenizer(gptTokenizerFile)
		if err != nil {
			fmt.Printf("Failed to load tokenizer from %s: %v\n", gptTokenizerFile, err)
			return
		}

		trainData, err := gpt.LoadTrainingData(gptInputFile)
		if err != nil {
			fmt.Printf("Failed to read training file %s: %v\n", gptInputFile, err)
			return
		}

		tokenIDs := vocab.Encode(string(trainData))
		if len(tokenIDs) < 2 {
			fmt.Println("Error: Tokenized corpus has fewer than 2 tokens.")
			return
		}

		vocabSize := len(vocab.GetData())
		dModel := 64
		numLayers := 2
		numHeads := 4
		scale := float32(0.02)

		fmt.Printf("Initializing Mini-GPT Model (vocabSize=%d, dModel=%d, layers=%d, heads=%d, maxSeqLen=%d)...\n",
			vocabSize, dModel, numLayers, numHeads, gptMaxSeqLen)

		model, err := gpt.NewGPTModel(vocabSize, dModel, numLayers, numHeads, gptMaxSeqLen, scale)
		if err != nil {
			fmt.Printf("Failed to create GPT model: %v\n", err)
			return
		}

		optimizer := gpt.NewAdamW(float32(gptLR), 0.9, 0.999, 1e-8, 0.01)

		contextSize := gptMaxSeqLen
		totalTokens := len(tokenIDs)
		stepsPerEpoch := (totalTokens - 1) / contextSize
		totalSteps := stepsPerEpoch * gptEpochs

		fmt.Println("----------------------------------------")
		fmt.Printf("Training corpus : %d tokens\n", totalTokens)
		fmt.Printf("Context size    : %d tokens\n", contextSize)
		fmt.Printf("Steps per epoch : ~%d\n", stepsPerEpoch)
		fmt.Printf("Total epochs    : %d\n", gptEpochs)
		fmt.Printf("Learning rate   : %g\n", gptLR)
		fmt.Println("----------------------------------------")
		fmt.Println("Training started!")
		fmt.Println()

		trainStart := time.Now()

		for epoch := 1; epoch <= gptEpochs; epoch++ {
			var totalLoss float32
			steps := 0
			epochStart := time.Now()

			for start := 0; start+contextSize < totalTokens; start += contextSize {
				end := start + contextSize + 1
				if end > totalTokens {
					break
				}
				chunk := tokenIDs[start:end]
				loss, err := model.TrainStep(chunk, optimizer)
				if err != nil {
					fmt.Printf("Epoch %d step %d failed: %v\n", epoch, steps, err)
					break
				}
				totalLoss += loss
				steps++

				if gptPrintEvery > 0 && steps%gptPrintEvery == 0 {
					elapsed := time.Since(epochStart).Seconds()
					stepsPerSec := float64(steps) / elapsed
					epochETA := float64(stepsPerEpoch-steps) / stepsPerSec

					completedTotal := (epoch-1)*stepsPerEpoch + steps
					overallElapsed := time.Since(trainStart).Seconds()
					overallETA := float64(totalSteps-completedTotal) / (float64(completedTotal) / overallElapsed)

					fmt.Printf("  epoch %d | step %d/%d | loss %.4f | %.2f steps/s | epoch ETA ~%.0fs | total ETA ~%.0fs\n",
						epoch, steps, stepsPerEpoch, loss, stepsPerSec, epochETA, overallETA)
				}
			}

			epochTime := time.Since(epochStart).Seconds()
			if steps > 0 {
				fmt.Printf("Epoch %d/%d done — avg loss: %.4f | steps: %d | time: %.1fs\n",
					epoch, gptEpochs, totalLoss/float32(steps), steps, epochTime)
			} else {
				fmt.Printf("Epoch %d/%d — no steps (corpus too short for context size %d)\n", epoch, gptEpochs, contextSize)
			}
			fmt.Println()
		}

		fmt.Printf("Training complete in %.1fs\n", time.Since(trainStart).Seconds())

		if err := gpt.SaveModel(model, gptOutputFile); err != nil {
			fmt.Printf("Failed to save GPT model to %s: %v\n", gptOutputFile, err)
			return
		}

		fmt.Printf("Model saved to %s\n", gptOutputFile)
	},
}

func init() {
	trainGptCmd.Flags().StringVarP(&gptInputFile, "input", "i", "train.txt", "Path to training text corpus file")
	trainGptCmd.Flags().StringVarP(&gptTokenizerFile, "tokenizer", "t", "tokenizer_model_data.json", "Path to tokenizer JSON file")
	trainGptCmd.Flags().StringVarP(&gptOutputFile, "output", "o", "gpt_model.json", "Path to save trained GPT model JSON")
	trainGptCmd.Flags().IntVarP(&gptEpochs, "epochs", "e", 5, "Number of training epochs")
	trainGptCmd.Flags().Float64VarP(&gptLR, "lr", "l", 0.001, "Learning rate for AdamW")
	trainGptCmd.Flags().IntVarP(&gptMaxSeqLen, "seq-len", "s", 64, "Maximum sequence / context length")
	trainGptCmd.Flags().IntVarP(&gptPrintEvery, "print-every", "p", 50, "Print progress every N steps (0 = epoch end only)")

	rootCmd.AddCommand(trainGptCmd)
}
