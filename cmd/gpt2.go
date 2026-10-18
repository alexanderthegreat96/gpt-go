package cmd

import (
	"fmt"
	"time"

	"github.com/alexanderthegreat96/gpt-go/gpt"
	"github.com/spf13/cobra"
)

var (
	gpt2WeightsFile      string
	gpt2NumHeads         int
	gpt2Prefix           string
	gpt2ReferenceFile    string
	gpt2Tolerance        float64
	gpt2RelTolerance     float64
	gpt2MeanRelTolerance float64
	gpt2ShowAll          bool
	gpt2InspectFull      bool
	gpt2Int8             bool
)

// short experiment command
// to see if gpt2 loads
var loadGpt2Cmd = &cobra.Command{
	Use:   "load-gpt2",
	Short: "Load a pretrained GPT-2 safetensors checkpoint and report what came out of it",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("Loading GPT-2 weights from %s\n", gpt2WeightsFile)
		fmt.Printf("Using Backend: %s\n", gpt.GetActiveBackend())

		start := time.Now()
		model, err := gpt.LoadGPT2(gpt2WeightsFile, gpt.GPT2Config{
			NumHeads: gpt2NumHeads,
			Prefix:   gpt2Prefix,
		})
		if err != nil {
			fmt.Printf("Failed to load: %v\n", err)
			return
		}

		vocabSize, dModel, numLayers, numHeads, maxSeqLen := model.Describe()

		fmt.Println("----------------------------------------")
		fmt.Printf("Vocab size      : %d\n", vocabSize)
		fmt.Printf("Hidden dim      : %d\n", dModel)
		fmt.Printf("Layers          : %d\n", numLayers)
		fmt.Printf("Heads           : %d (head dim %d)\n", numHeads, dModel/numHeads)
		fmt.Printf("Context length  : %d\n", maxSeqLen)
		fmt.Printf("Parameters      : %s\n", gpt.FormatParamCount(model.CountParameters()))
		if model.IsLMHeadTied() {
			// worth saying out loud -> the count above treats the tied head as
			// the same weights as the embeddings, but memory does not
			fmt.Printf("LM head         : tied to the token embeddings (held as a transposed copy)\n")
		}
		fmt.Printf("Load time       : %.1fs\n", time.Since(start).Seconds())
		fmt.Println("----------------------------------------")

		fmt.Println()
		fmt.Println("Weights are in. Do NOT trust the output yet -> run verify-gpt2")
		fmt.Println("against a reference dump first. A transposed matrix or the wrong")
		fmt.Println("GELU produces fluent nonsense rather than an error.")
	},
}

var verifyGpt2Cmd = &cobra.Command{
	Use:   "verify-gpt2",
	Short: "Compare our forward pass against reference activations dumped from PyTorch",
	Long: "Loads a GPT-2 checkpoint, runs the token sequence stored inside the reference\n" +
		"dump through our forward pass, and compares every intermediate activation.\n\n" +
		"Generate the reference with: python scripts/dump_gpt2_reference.py",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("Weights   : %s\n", gpt2WeightsFile)
		fmt.Printf("Reference : %s\n", gpt2ReferenceFile)
		fmt.Printf("Backend   : %s\n", gpt.GetActiveBackend())
		fmt.Printf("Tolerance : abs %g | max rel %g | mean rel %g\n",
			gpt2Tolerance, gpt2RelTolerance, gpt2MeanRelTolerance)

		model, err := gpt.LoadGPT2(gpt2WeightsFile, gpt.GPT2Config{
			NumHeads: gpt2NumHeads,
			Prefix:   gpt2Prefix,
			Quantize: gpt2Int8,
		})
		if err != nil {
			fmt.Printf("Failed to load weights: %v\n", err)
			return
		}

		tol := gpt.Tolerance{
			Abs:     float32(gpt2Tolerance),
			MaxRel:  float32(gpt2RelTolerance),
			MeanRel: float32(gpt2MeanRelTolerance),
		}

		gpt.ReportQuantization(model)

		diffs, tokenIDs, allPassed, err := gpt.VerifyAgainstReference(model, gpt2ReferenceFile, tol)
		if err != nil {
			fmt.Printf("Verification failed to run: %v\n", err)
			return
		}

		fmt.Printf("Tokens    : %v (%d of them)\n", tokenIDs, len(tokenIDs))
		fmt.Println()
		fmt.Printf("%-14s %-8s %11s %11s %11s %11s %10s\n",
			"TENSOR", "STATUS", "MAX DIFF", "MEAN DIFF", "MAX REL", "MEAN REL", "SCALE")
		fmt.Println("----------------------------------------------------------------------------------------")

		// first failure is more important than others here
		// if we get a first failure, then everything else
		// is just busted
		firstFailure := ""

		for _, diff := range diffs {
			passed := diff.Passed(tol)
			if !passed && firstFailure == "" {
				firstFailure = diff.Name
			}

			// once something has gone wrong the rest is noise, so by default we
			// only print up to the first failure plus whatever failed after it
			if !gpt2ShowAll && firstFailure != "" && passed {
				continue
			}

			status := "ok"
			switch {
			case diff.Status == "missing":
				status = "skipped"
			case diff.Status == "shape":
				status = "SHAPE"
			case !passed:
				status = "FAIL"
			}

			if diff.Status == "missing" {
				fmt.Printf("%-14s %-8s %11s %11s %11s %11s %10s\n", diff.Name, status, "-", "-", "-", "-", "-")
				continue
			}

			if diff.Status == "shape" {
				fmt.Printf("%-14s %-8s  got %dx%d, want %dx%d\n",
					diff.Name, status, diff.GotRows, diff.GotCols, diff.RefRows, diff.RefCols)
				continue
			}

			// SCALE is the largest magnitude in the reference tensor. it is the
			// column that makes the absolute diffs mean anything -> 7e-3 against
			// a scale of 3308 is the eighth digit, not a bug
			fmt.Printf("%-14s %-8s %11.3e %11.3e %11.2e %11.2e %10.1f\n",
				diff.Name, status, diff.MaxAbsDiff, diff.MeanAbsDiff,
				diff.MaxRelDiff, diff.MeanRelDiff, diff.RefMaxAbs)
		}

		fmt.Println("----------------------------------------------------------------------------------------")

		if allPassed {
			fmt.Println()
			fmt.Println("Everything matches. The weights are loaded correctly and the forward")
			fmt.Println("pass agrees with PyTorch -> you can trust the output now.")
			return
		}

		fmt.Println()
		fmt.Printf("First divergence: %s\n", firstFailure)
		fmt.Println()
		fmt.Println("Fix that one and re-run. Everything after it in the list is just the")
		fmt.Println("same error flowing downstream. Usual suspects, roughly in order:")
		fmt.Println("  - MEAN REL near the noise floor (1e-6ish) while MAX REL is fine too")
		fmt.Println("    is not a bug at all, it is float32 on a big SCALE. widen the")
		fmt.Println("    tolerances rather than hunting for something that is not there")
		fmt.Println("  - a shape mismatch means a transpose, a bad c_attn split or a")
		fmt.Println("    reshape that lost a dimension (a whole-model size mismatch is")
		fmt.Println("    caught before we get this far, so it is not that)")
		fmt.Println("  - a diff that starts small and grows per layer means the wrong GELU")
		fmt.Println("  - ln_1 or ln_2 off immediately means swapped gamma/beta or epsilon")
		fmt.Println("  - attn off but ln_1 fine means the head layout or the qkv split")
		fmt.Println("  - everything fine until logits means the lm head orientation")
	},
}

var inspectSafeTensorsCmd = &cobra.Command{
	Use:   "inspect-safetensors",
	Short: "List the tensors in a safetensors file with their shapes and dtypes",
	Run: func(cmd *cobra.Command, args []string) {
		st, err := gpt.OpenSafeTensors(gpt2WeightsFile)
		if err != nil {
			fmt.Printf("Failed to open: %v\n", err)
			return
		}
		defer st.Close()

		names := st.Names()

		fmt.Printf("File    : %s\n", gpt2WeightsFile)
		fmt.Printf("Tensors : %d\n", len(names))

		if len(st.Metadata) > 0 {
			fmt.Println("Metadata:")
			for key, value := range st.Metadata {
				// the token list can be long, so keep it on one line
				if len(value) > 100 {
					value = value[:97] + "..."
				}
				fmt.Printf("  %-12s %s\n", key, value)
			}
		}

		fmt.Println()

		if gpt2InspectFull {
			for _, name := range names {
				shape, err := st.Shape(name)
				if err != nil {
					fmt.Printf("  %-44s %v\n", name, err)
					continue
				}
				fmt.Printf("  %-44s %v\n", name, shape)
			}
			return
		}

		// the default is a summary, because a full gpt-2 xl listing is 600
		// lines and the thing you actually want to know is the model's shape
		shown := 0
		for _, name := range names {
			if !gpt.IsShapeDefiningTensor(name) {
				continue
			}
			shape, err := st.Shape(name)
			if err != nil {
				continue
			}
			fmt.Printf("  %-44s %v\n", name, shape)
			shown++
		}

		fmt.Println()
		fmt.Printf("  (%d of %d tensors shown, pass --full to list everything)\n", shown, len(names))
	},
}

func init() {
	loadGpt2Cmd.Flags().StringVarP(&gpt2WeightsFile, "weights", "w", "models/gpt2/model.safetensors", "Path to the GPT-2 safetensors checkpoint")
	loadGpt2Cmd.Flags().IntVar(&gpt2NumHeads, "heads", 0, "Attention head count (0 = infer from the hidden dim)")
	loadGpt2Cmd.Flags().StringVar(&gpt2Prefix, "prefix", "", "Tensor name prefix (empty = auto-detect, usually \"\" or \"transformer.\")")
	loadGpt2Cmd.Flags().BoolVar(&gpt2Int8, "int8", false, "Quantize the matmul weights to int8 and report the saving")

	verifyGpt2Cmd.Flags().StringVarP(&gpt2WeightsFile, "weights", "w", "models/gpt2/model.safetensors", "Path to the GPT-2 safetensors checkpoint")
	verifyGpt2Cmd.Flags().StringVarP(&gpt2ReferenceFile, "reference", "r", "models/gpt2/reference_activations.safetensors", "Path to the reference activation dump")
	verifyGpt2Cmd.Flags().IntVar(&gpt2NumHeads, "heads", 0, "Attention head count (0 = infer from the hidden dim)")
	verifyGpt2Cmd.Flags().StringVar(&gpt2Prefix, "prefix", "", "Tensor name prefix (empty = auto-detect)")
	verifyGpt2Cmd.Flags().Float64Var(&gpt2Tolerance, "tolerance", 1e-3, "Absolute difference accepted outright, for tensors too small for a relative figure to mean anything")
	verifyGpt2Cmd.Flags().Float64Var(&gpt2RelTolerance, "rel-tolerance", 1e-3, "Max relative difference accepted (worst element, against the tensor's largest magnitude)")
	verifyGpt2Cmd.Flags().Float64Var(&gpt2MeanRelTolerance, "mean-rel-tolerance", 1e-4, "Max mean relative difference accepted -> this is the one that catches a wrong activation or a missing bias")
	verifyGpt2Cmd.Flags().BoolVar(&gpt2ShowAll, "show-all", false, "Print every tensor instead of stopping the noise after the first failure")
	verifyGpt2Cmd.Flags().BoolVar(&gpt2Int8, "int8", false, "Quantize the matmul weights to int8 before comparing, to measure what that costs in accuracy")

	inspectSafeTensorsCmd.Flags().StringVarP(&gpt2WeightsFile, "file", "f", "models/gpt2/model.safetensors", "Path to any safetensors file (weights or a reference dump)")
	inspectSafeTensorsCmd.Flags().BoolVar(&gpt2InspectFull, "full", false, "List every tensor instead of just the ones that define the model shape")

	rootCmd.AddCommand(loadGpt2Cmd)
	rootCmd.AddCommand(verifyGpt2Cmd)
	rootCmd.AddCommand(inspectSafeTensorsCmd)
}
