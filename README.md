# gpt-go

A GPT-2 style language model built from scratch in pure Go — no machine learning frameworks, no BLAS, no autograd. Every component including the tokenizer, matrix operations, forward pass, full backpropagation, and AdamW optimizer is implemented by hand.

---

## Architecture Overview

```
Input text
    │
    ▼
┌─────────────────────┐
│   BPE Tokenizer     │  byte-pair encoding, learned merges
└─────────────────────┘
    │  token IDs []int
    ▼
┌─────────────────────┐
│  Token Embedding    │  [vocabSize × dModel]  lookup table
│  Pos. Embedding     │  [maxSeqLen × dModel]  learned positions
│       (+)           │  element-wise sum
└─────────────────────┘
    │  [seqLen × dModel]
    ▼
┌─────────────────────┐  ×numLayers
│  Transformer Block  │
│  ┌───────────────┐  │
│  │  LayerNorm 1  │  │
│  │  Multi-Head   │  │  causal self-attention, h heads
│  │  Attention    │  │  Q, K, V projections + output projection
│  │  + residual   │  │
│  ├───────────────┤  │
│  │  LayerNorm 2  │  │
│  │  FFN          │  │  Linear → GELU → Linear (4× expansion)
│  │  + residual   │  │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Final LayerNorm    │
│  LM Head            │  [dModel × vocabSize]  linear projection
└─────────────────────┘
    │  logits [seqLen × vocabSize]
    ▼
┌─────────────────────┐
│  Softmax + Sample   │  temperature, top-k, repetition penalty
└─────────────────────┘
    │  next token
    ▼
  output text
```

---

## Components

### BPE Tokenizer (`bpe/`)

Byte-Pair Encoding tokenizer trained directly on the corpus.

The implementation is snatched from:
[byte-pair-encoding](https://github.com/alexanderthegreat96/byte-pair-encoding)

- Starts with a base vocabulary of 256 byte tokens
- Iteratively merges the most frequent adjacent token pair
- Stores an ordered merge list so encoding is deterministic
- Parallel pair counting via goroutines for large corpora
- In-place merge application (no allocation per iteration)

**Files:**

- `tokenizer.go` — `Vocabulary`, `Encode`, `Decode`, `TrainFromBytes`, `TrainFromString`
- `persistence.go` — `SaveTokenizer`, `LoadTokenizer` (JSON)

---

### Matrix (`gpt/matrix.go`)

All tensor operations are `float32` row-major matrices.

| Method              | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `MatrixMul`         | delegates to active backend (default: ikj cache-friendly loop) |
| `MatrixMulBackward` | delegates to active backend; returns dA = dC·Bᵀ and dB = Aᵀ·dC |
| `MatrixAdd`         | element-wise addition                                          |
| `Transpose`         | returns new transposed matrix                                  |
| `Scale`             | in-place scalar multiply                                       |
| `Randomize(scale)`  | uniform random init in `[-scale, scale]`                       |
| `AccumulateGrad`    | adds gradient into `grad` buffer                               |
| `ZeroGrad`          | zeros `grad` buffer                                            |
| `ConcatRows`        | vertical concatenation (used by KV cache)                      |
| `GetRow`            | returns a copy of one row                                      |

---

### Matrix Multiply Backend (`gpt/backend.go`, `gpt/backend_blas.go`)

Matrix multiply dominates both the forward and backward pass — every attention projection (Q, K, V, O), every FFN layer, and the LM head are matmuls. The implementation is abstracted behind a `Backend` interface so the underlying kernel can be swapped without touching any model code.

```go
type Backend interface {
    MatMul(a, b *Matrix) *Matrix        // C = A · B
    MatMulTransB(a, b *Matrix) *Matrix  // C = A · Bᵀ  (used for dA in backward)
    MatMulTransA(a, b *Matrix) *Matrix  // C = Aᵀ · B  (used for dB in backward)
}
```

**GoBackend** (default, always compiled) — pure Go with cache-friendly loop ordering. No dependencies.

**BlasBackend** (opt-in) — delegates to `gonum/blas32`, which is backed by OpenBLAS. Uses hand-written AVX/AVX-512 assembly, cache tiling, and FMA instructions. A naive triple loop reaches ~5–10% of peak CPU throughput; OpenBLAS reaches ~80–90%. The speedup is meaningful once `dModel ≥ 256`.

Swapping backends is one call at startup:

```go
gpt.SetBackend(&gpt.BlasBackend{})
```

The BLAS backend is gated behind a build tag so the default binary stays dependency-free. See the [BLAS backend](#blas-backend) section under Usage for build instructions.

---

### Activations (`gpt/activations.go`)

| Function          | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `GELU`            | `0.5 * x * (1 + erf(x / √2))` — smoother than ReLU, standard in GPT |
| `GELUBackward`    | analytical derivative of GELU                                       |
| `Softmax`         | numerically stable row-wise softmax (subtracts row max)             |
| `SoftmaxBackward` | Jacobian-vector product: `p * (dy - dot(dy, p))`                    |

---

### Layer Normalization (`gpt/normalization.go`)

`LayerNorm(gamma, beta, epsilon)` — normalizes across the feature dimension (per token row):

```
xHat = (x - mean) / sqrt(var + ε)
y    = gamma * xHat + beta
```

`LayerNormBackward` computes dInput, dGamma, dBeta using the full derivation through variance and mean.

---

### Embeddings (`gpt/embeddings.go`)

**Token Embedding** — `[vocabSize × dModel]` lookup table. `Forward(tokenIDs)` gathers rows.

**Positional Embedding** — `[maxSeqLen × dModel]` learned table. `ForwardAt(seqLen, offset)` slices starting at `offset`, which is used during cached generation so each new token gets the correct position.

---

### Causal Self-Attention (`gpt/attention.go`)

Multi-head causal (decoder-only) attention.

```
Q = input · Wq
K = input · Wk
V = input · Wv

For each head h:
    scores[i,j] = Q[i,h] · K[j,h] / sqrt(headDim)    (masked: j > i → -1e9)
    probs        = softmax(scores)
    out[i,h]     = Σⱼ probs[i,j] · V[j,h]

output = concat(heads) · Wo
```

- `headDim = dModel / numHeads`
- Inference (`Forward`) runs heads in parallel via goroutines
- Training (`ForwardSave`) runs sequentially and saves all intermediates needed for backward
- `Backward` propagates through Wo, the softmax, and the Q/K/V projections

**KV Cache** — `ForwardWithCache` appends new K/V rows to the cache at each generation step, avoiding recomputation of past keys and values.

---

### Feed-Forward Network (inside `gpt/transformer.go`)

Two-layer MLP with 4× inner expansion:

```
FFN(x) = GELU(x · W1) · W2
```

`dModel → 4*dModel → dModel`

---

### Transformer Block (`gpt/transformer.go`)

Pre-LN architecture (LayerNorm before each sub-layer, not after):

```
x = x + Attention(LayerNorm1(x))
x = x + FFN(LayerNorm2(x))
```

Backward splits the residual gradient: both the skip path and the sub-layer path receive the full upstream gradient.

---

### GPT Model (`gpt/model.go`)

Assembles all components:

```go
NewGPTModel(vocabSize, dModel, numLayers, numHeads, maxSeqLen int, scale float32)
```

| Method                   | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `Forward`                | inference-only forward pass                                                        |
| `ForwardWithCache`       | autoregressive inference with KV cache                                             |
| `ForwardTrain`           | forward pass that saves activations for backprop                                   |
| `Backward`               | full backpropagation through LM head → final LN → blocks → embeddings              |
| `TrainStep`              | one full step: zero grads → forward → cross-entropy loss → backward → AdamW update |
| `GetTrainableParameters` | returns all parameter matrices                                                     |

---

### Loss (`gpt/loss.go`)

`CrossEntropyLoss(logits, targetID)` — computes softmax, then `-log(p[target])`. Returns both the scalar loss and the gradient `(p - one_hot(target))`.

---

### AdamW Optimizer (`gpt/optimizer.go`)

```
m = β₁·m + (1 - β₁)·g
v = β₂·v + (1 - β₂)·g²
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
θ = θ - lr · weight_decay · θ   (decoupled weight decay)
θ = θ - lr · m̂ / (√v̂ + ε)
```

Default: `β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=0.01`

---

### Generation (`gpt/generate.go`)

```go
type GenerateConfig struct {
    Temperature       float32   // 0 = greedy argmax
    TopK              int       // 0 = full distribution
    RepetitionPenalty float32   // 1.0 = none; >1 discourages repeats
}
```

Sampling pipeline per step:

1. Apply repetition penalty (CTRL-style: divide positive logits, multiply negative)
2. Divide by temperature
3. Mask all but top-k logits to `-1e9`
4. Softmax → sample via cumulative distribution

---

### Persistence (`gpt/persistence.go`, `bpe/persistence.go`)

Both the tokenizer and model serialize to JSON. Model weights are stored as flat `float32` arrays with shape metadata.

---

---

### BLAS Backend

To enable the BLAS backend:

```bash
go get gonum.org/v1/gonum
```

Then build or run with the `blas` tag:

```bash
go build -tags blas -o gpt-go .
go run -tags blas main.go train-gpt ...
```

And activate it in code before training or generation starts:

```go
gpt.SetBackend(&gpt.BlasBackend{})
```

Without the tag, `BlasBackend` does not exist and the binary has no external ML dependencies.

---

## Project Structure

```
gpt-go/
├── main.go
├── cmd/
│   ├── root.go          — cobra root command
│   ├── train_bpe.go     — CLI: train BPE tokenizer
│   ├── train_gpt.go     — CLI: train GPT model
│   └── run.go           — CLI: text generation
├── bpe/
│   ├── tokenizer.go
│   ├── persistence.go
│   └── tokenizer_test.go
└── gpt/
    ├── matrix.go
    ├── activations.go
    ├── normalization.go
    ├── embeddings.go
    ├── attention.go
    ├── transformer.go
    ├── model.go
    ├── generate.go
    ├── optimizer.go
    ├── loss.go
    ├── persistence.go
    └── *_test.go
```

---

## Usage

### 1. Train the BPE tokenizer

```bash
go run main.go train-bpe --input train.txt --output tokenizer_model_data.json --vocab-size 2000
```

| Flag                  | Default                     | Description            |
| --------------------- | --------------------------- | ---------------------- |
| `--input` / `-i`      | `train.txt`                 | Training corpus        |
| `--output` / `-o`     | `tokenizer_model_data.json` | Output file            |
| `--vocab-size` / `-v` | `2000`                      | Target vocabulary size |

---

### 2. Train the GPT model

```bash
go run main.go train-gpt \
  --input train.txt \
  --tokenizer tokenizer_model_data.json \
  --output gpt_model.json \
  --epochs 5 \
  --lr 0.001 \
  --seq-len 64 \
  --print-every 50
```

| Flag                   | Default                     | Description                                       |
| ---------------------- | --------------------------- | ------------------------------------------------- |
| `--input` / `-i`       | `train.txt`                 | Training corpus                                   |
| `--tokenizer` / `-t`   | `tokenizer_model_data.json` | Tokenizer file                                    |
| `--output` / `-o`      | `gpt_model.json`            | Where to save the trained model                   |
| `--epochs` / `-e`      | `5`                         | Number of passes over the corpus                  |
| `--lr` / `-l`          | `0.001`                     | AdamW learning rate                               |
| `--seq-len` / `-s`     | `64`                        | Context window size (tokens)                      |
| `--print-every` / `-p` | `50`                        | Print progress every N steps (0 = epoch end only) |

**Hardcoded model config:** `dModel=64, numLayers=2, numHeads=4, scale=0.02`

**Progress output:**

```
epoch 1 | step 50/5694 | loss 6.8102 | 19.79 steps/s | epoch ETA ~285s | total ETA ~1436s
```

---

### 3. Generate text

```bash
go run main.go run \
  --model gpt_model.json \
  --tokenizer tokenizer_model_data.json \
  --prompt "To be or not" \
  --tokens 100 \
  --temperature 0.8 \
  --top-k 40 \
  --rep-penalty 1.2
```

| Flag                 | Default                     | Description                       |
| -------------------- | --------------------------- | --------------------------------- |
| `--prompt` / `-p`    | `"Hello world"`             | Prompt string                     |
| `--tokenizer` / `-t` | `tokenizer_model_data.json` | Tokenizer file                    |
| `--model` / `-m`     | `gpt_model.json`            | Model file                        |
| `--tokens` / `-n`    | `8`                         | New tokens to generate            |
| `--temperature`      | `1.0`                       | Sampling temperature (0 = greedy) |
| `--top-k`            | `40`                        | Top-k pool size (0 = full vocab)  |
| `--rep-penalty`      | `1.0`                       | Repetition penalty (1.0 = off)    |

---

## Recommended settings for tinyshakespeare

**Tokenizer:**

```bash
go run main.go train-bpe --input train.txt --vocab-size 2000
```

**Training:**

```bash
go run main.go train-gpt --input train.txt --tokenizer tokenizer_model_data.json --epochs 10 --lr 0.001 --seq-len 64 --print-every 50
```

**Generation:**

```bash
go run main.go run --prompt "KING:" --tokens 200 --temperature 0.8 --top-k 40 --rep-penalty 1.2
```

At ~20 steps/sec on a modern CPU, 5 epochs over tinyshakespeare (~5700 steps/epoch) takes roughly 25 minutes. Loss should drop from ~7.6 (random) toward 5–6 within the first epoch.

---

## Dependencies

| Dependency               | When                   |
| ------------------------ | ---------------------- |
| `github.com/spf13/cobra` | always — CLI framework |
| `gonum.org/v1/gonum`     | only with `-tags blas` |

Default build has no external ML dependencies.

---

## Credits

Now, I did not study machine learning and advanced math from scratch in a couple of days, that would be impossible. Therefore, I leveraged Gemini to help me understand the architecture and concepts, while I wrote the code. Claude was used at the end to find some critical bugs and flaws I introduced.

| Role                      | Who    |
| ------------------------- | ------ |
| Go Code                   | me     |
| Concept guidance & theory | Gemini |
| Bug fixes & improvements  | Claude |
