# gpt-go

A GPT-2 style language model built from scratch in pure Go — no machine learning frameworks, no cgo, no autograd. Every component including the tokenizer, matrix operations, forward pass, full backpropagation, and AdamW optimizer is implemented by hand.

It does two things:

**[Training](#training)** — a small GPT trained from scratch on your own corpus, with a BPE tokenizer trained alongside it. Good for learning how the pieces fit together; too small to know any facts.

**[Running](#running)** — generate text, either from a model you trained or from a real pretrained GPT-2. OpenAI's published weights load through this same transformer, any of the four sizes up to 1.5B parameters, using GPT-2's own byte-level BPE. The forward pass is verified numerically against PyTorch rather than assumed correct.

The second mode exists because of the arithmetic: training GPT-2 from scratch on a CPU is roughly `6 × params × tokens` ≈ 7.4 × 10¹⁸ FLOPs, which is years. Inference is `2 × params` per token — about 10,000× less work, and entirely practical.

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

The BLAS backend is gated behind a build tag so the default binary stays dependency-free. See [BLAS Backend](#blas-backend) for build instructions.

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

    StopTokenIDs []int          // nil = never stop early
    SlideContext bool           // false = stop at maxSeqLen
    OnToken      func(id int)   // nil = no streaming callback
}
```

Sampling pipeline per step:

1. Apply repetition penalty (CTRL-style: divide positive logits, multiply negative)
2. Divide by temperature
3. Mask all but top-k logits to `-1e9`
4. Softmax → sample via cumulative distribution

Every field's zero value preserves the original behaviour, so existing callers are unaffected. `StopTokenIDs` is a slice rather than an `int` deliberately — `nil` means "never stop", whereas an `int` field would have made token id 0 a stop token by accident, and 0 is a real id in a byte-level vocabulary.

**`SlideContext`** keeps generating past the context window. It cannot simply drop the oldest cache rows: positional embeddings are absolute and already baked into the cached keys, so the survivors would keep their original positions while new tokens got numbered from the shortened length, putting two tokens at the same position. Instead it retains the most recent half, clears the caches, and re-runs a forward pass over the retained tokens to renumber them from zero — one prefill every `maxSeqLen/2` tokens.

**`OnToken`** receives token **ids**, not text. Under byte-level BPE a single token can be a fragment of a UTF-8 sequence, so a caller that prints incrementally must buffer bytes and emit only complete characters. `cmd/gpt2_repl.go` shows how.

---

### Persistence (`gpt/persistence.go`, `bpe/persistence.go`)

Both the tokenizer and model serialize to JSON. Model weights are stored as flat `float32` arrays with shape metadata.

JSON costs about 11 bytes per parameter, which is fine for the models trained here and hopeless for anything larger — GPT-2 small would be a 1.4 GB file to parse on every run. Pretrained weights therefore use safetensors instead (below), and `SaveModel` refuses outright to serialize an int8-quantized model rather than writing a file full of nulls that would load without complaint.

Bias terms and the activation choice are stored with `omitempty`, so a model file written before those existed still loads byte-identically.

---

### Safetensors Reader (`gpt/safetensors.go`)

The format HuggingFace publishes weights in, and about as simple as a binary format gets: 8 bytes of little-endian header length, that many bytes of JSON mapping tensor name to dtype/shape/byte-range, then raw tensor bytes. The one trap is that those byte ranges are relative to the **start of the data section**, not the start of the file.

Tensors are read on demand with `ReadAt`, in bounded 256K-element chunks, rather than loading the file into memory. Holding the whole file *and* the matrices built from it doubled peak memory — 3.9 GB against a 1.5 GB file — and the file buffer existed only to be copied out of.

mmap would be the fancier answer and is not better here: the QKV split and the lm_head transpose get materialized either way, so a mapping of the whole file would sit *on top of* those rather than instead of them. It also needs `unsafe` and per-platform build tags. Measured projections put streaming at 1.51 GB for gpt2-medium versus 1.79 GB for zero-copy mmap.

Handles `F32`, `F16` and `BF16`. BF16 widening is a 16-bit shift; F16 needs real conversion, including the subnormal case.

---

### Pretrained GPT-2 Loader (`gpt/gpt2.go`)

Maps a HuggingFace checkpoint onto `GPTModel`. This works because the architecture here *is* GPT-2's: pre-LN blocks, GELU, 4× inner FFN, learned positional embeddings, heads laid out contiguously across the hidden dimension.

Five things differ, and all five are handled at load time:

| | |
| --- | --- |
| **Bias terms** | GPT-2 has four per block; the from-scratch model has none. Nil means "skip", so untrained models are unaffected |
| **GELU flavour** | GPT-2 was trained on the tanh approximation (`gelu_new`), not the erf form |
| **Fused `c_attn`** | Q, K and V share one `(d, 3d)` matrix, split by column range |
| **Tied lm head** | Tied to `wte`, which is stored transposed — the one unavoidable transpose |
| **File format** | safetensors, not JSON |

Weight orientation needs no transposing at all: HuggingFace GPT-2 uses `Conv1D`, whose weight is `(in, out)` with a forward pass of `x @ w + b` — exactly this codebase's `input.MatrixMul(weight)` convention. Had OpenAI used `nn.Linear`, every matrix would need flipping.

Model shape is read off the weights rather than trusted from a config, so one code path handles all four sizes. The head count cannot be inferred from the file — nothing in it says how the hidden dimension is sliced — so it comes from a table of the published sizes, and an unrecognized width is an error rather than a guess.

**A mis-loaded model does not error.** It produces fluent, grammatical nonsense indistinguishable from an undertrained model. Hence the next component.

---

### Reference Verification (`gpt/verify.go`)

The correctness gate. A Python script dumps intermediate activations from PyTorch for a fixed prompt; this compares our forward pass against them, tensor by tensor, in network order. Instead of "the output is garbage" you get "block 3 attention diverges, everything before it matches".

Capture points, 6 per block plus 3: embeddings, each `ln_1`, attention output, first residual, each `ln_2`, MLP output, block output, `ln_f`, logits. `CollectActivations` reuses the `BlockActivations` that `ForwardTrain` already stashes for backprop, so there is no second forward implementation to drift out of sync.

Judgement is **relative**, on two numbers that fail for different reasons:

- **`MaxRel`** — worst element against the tensor's largest magnitude. Catches a few badly wrong values.
- **`MeanRel`** — mean error against mean magnitude. Catches error smeared across everything, which is what a wrong activation function or a missing bias actually looks like.

Both must hold. A single absolute threshold does not work: GPT-2's activations span five orders of magnitude, from 2.5 at the embeddings to 3308 in block 3's MLP output because of the outlier features the architecture is known for. float32 carries ~1e-3 of rounding on a value that large before any accumulation, so an absolute tolerance demands precision float32 does not have.

The tolerances are calibrated against a real gpt2-medium run: worst observed was 2.9e-5 max relative and ~1e-5 mean relative, so the defaults sit an order of magnitude above that noise floor and an order of magnitude below what a real bug produces — swapping the tanh GELU for erf moves mean relative error to ~1e-3.

Before comparing anything it cross-checks that the two files describe the same model, by layer count and hidden size. Pairing gpt2-medium weights with a gpt2-small reference otherwise produces a wall of shape errors that reads exactly like a transpose bug.

---

### GPT-2 Byte-Level BPE (`bpe/gpt2.go`)

A separate tokenizer from the trainable one in `bpe/tokenizer.go`, because the 50257 embedding rows in a pretrained checkpoint are indexed by *OpenAI's* token ids. One token out of place and the model answers a different prompt than you typed.

Three things make it awkward:

1. **Byte level.** Arbitrary bytes must survive a round trip, but BPE works on text. Every byte maps through a fixed 256-entry table to a printable codepoint — which is why a space appears as `Ġ` (byte `0x20` → `U+0120`) throughout `vocab.json`.
2. **Ranked merges.** `merges.txt` is ordered and the line number is the priority. Each step merges the **lowest-ranked** adjacent pair in the whole word, not the first one found.
3. **A pre-tokenizer regex Go cannot compile.** The original pattern ends `\s+(?!\S)`, and `(?!` is a negative lookahead — Go's `regexp` is RE2 and rejects it outright. So the scan is hand-rolled.

That last point matters more than it looks. The lookahead means that in a whitespace run followed by a word, the **last space detaches and joins the word**:

```
"hello  world"  →  ["hello", " ", " world"]      correct
                →  ["hello", "  ", "world"]      without the lookahead
```

Different pre-tokens, different ids, a silently different prompt. Correctness is pinned by fixtures generated from the real tokenizer — 28 cases plus a round trip over all 256 byte values — because a wrong tokenizer still emits perfectly valid ids and there is nothing to notice by eye.

`<|endoftext|>` (id 50256) is matched before the pre-tokenizer, matching HuggingFace. OpenAI's original `encoder.py` did not special-case it.

---

### Int8 Quantization (`gpt/quantize.go`)

Weight-only quantization with per-column scales, for when memory is the binding constraint. For `y = x @ W` with `W` shaped `(in, out)`, each output column gets its own scale:

```
W[i][j] ≈ scale[j] * q[i][j]          q in [-127, 127]
y[r][j] =  scale[j] * Σᵢ x[r][i] * q[i][j]
```

The scale factors out of the sum, so the inner loop stays a plain multiply-accumulate and the scale applies once per output element.

Per **column** rather than per tensor matters: one scale for the whole matrix has to cover the largest weight anywhere in it, and with GPT-2's outliers most columns would end up using a sliver of the int8 range.

The quantized data lives on `Matrix` itself, so `MatrixMul` is the only place that knows — every `input.MatrixMul(weight)` call site is unchanged. Quantization happens **during** the load, per tensor, not after: doing it afterwards leaves the steady state smaller but makes peak memory *worse*, since the float32 weights and their int8 copies are briefly both live (measured: 1,862 MB → 2,169 MB).

Not quantized, because they are not matmul operands: the token embeddings (a row lookup), layernorm gammas and betas (read per element), and every bias (added, not multiplied).

The cost is speed. gonum's float32 matmul is hand-written assembly; this is a Go loop. See [Performance and memory](#performance-and-memory).

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

Nothing else is needed — `backend_blas.go` has an `init()` that registers itself, so building with the tag is the whole switch. `gpt.SetBackend` exists if you want to choose explicitly, but the tag already does it.

Without the tag, `BlasBackend` does not exist and the binary has no external ML dependencies.

All the GPT-2 Makefile targets build with `-tags blas`, since it is several times faster and still cgo-free.

---

## Project Structure

```
gpt-go/
├── main.go
├── Makefile              — every workflow below, as a target
├── cmd/
│   ├── root.go           — cobra root command
│   ├── train_bpe.go      — CLI: train BPE tokenizer
│   ├── train_gpt.go      — CLI: train GPT model
│   ├── run.go            — CLI: generate from a model we trained
│   ├── gpt2.go           — CLI: load-gpt2, verify-gpt2, inspect-safetensors
│   ├── gpt2_run.go       — CLI: generate from a pretrained GPT-2
│   └── gpt2_repl.go      — CLI: interactive loop, weights loaded once
├── bpe/
│   ├── tokenizer.go      — trainable BPE
│   ├── gpt2.go           — GPT-2's byte level BPE
│   ├── persistence.go
│   ├── testdata/         — fixtures generated from the real tokenizer
│   └── *_test.go
├── gpt/
│   ├── matrix.go
│   ├── activations.go    — GELU (erf and tanh), softmax
│   ├── normalization.go
│   ├── embeddings.go
│   ├── attention.go
│   ├── transformer.go
│   ├── model.go
│   ├── generate.go
│   ├── optimizer.go
│   ├── loss.go
│   ├── persistence.go    — JSON, for models we train
│   ├── safetensors.go    — streaming reader for pretrained weights
│   ├── gpt2.go           — pretrained GPT-2 loader
│   ├── verify.go         — activation comparison against PyTorch
│   ├── quantize.go       — int8 weights
│   └── *_test.go
├── scripts/
│   ├── setup_gpt2.sh     — one command: download, check, verify
│   ├── dump_gpt2_reference.py
│   ├── dump_gpt2_tokenizer_fixtures.py
│   └── requirements.txt  — python deps, scripts only
└── models/
    ├── tiny_shakespeare/ — corpus + trained model, committed
    ├── tiny_wiki/        — corpus + trained model, committed
    └── gpt2*/            — downloaded checkpoints, gitignored
```

---

## Training

Training a model from scratch on your own corpus: a BPE tokenizer, then the model itself. Two commands, in that order, because the model needs the tokenizer's vocabulary size.

To run a model rather than train one — whether yours or a pretrained GPT-2 — see [Running](#running).

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

### Worked example: tinyshakespeare

```bash
go run main.go train-bpe --input train.txt --vocab-size 2000
go run main.go train-gpt --input train.txt --tokenizer tokenizer_model_data.json \
  --epochs 10 --lr 0.001 --seq-len 64 --print-every 50
```

At ~20 steps/sec on a modern CPU, 5 epochs over tinyshakespeare (~5700 steps/epoch) takes roughly 25 minutes. Loss should drop from ~7.6 (random) toward 5–6 within the first epoch.

Then [run it](#a-model-you-trained).

### What a model this size can do

Learn grammar and local style, and nothing else. At the hardcoded `dModel=64, numLayers=2` it has ~0.36M parameters — GPT-2 small has 124M and still invents facts. Expect plausible prose with no knowledge behind it.

If you want answers rather than prose, don't train for longer: the limit is capacity, not epochs. Use [a pretrained checkpoint](#a-pretrained-gpt-2) instead.

---

## Running

Two kinds of model, two sets of commands:

| | Command | Tokenizer |
| --- | --- | --- |
| **[A model you trained](#a-model-you-trained)** | `run` | your BPE, from `train-bpe` |
| **[A pretrained GPT-2](#a-pretrained-gpt-2)** | `gpt2-run`, `gpt2-repl` | GPT-2's byte-level BPE, 50257 tokens |

They are separate commands because the tokenizers are incompatible: a pretrained checkpoint's 50257 embedding rows are indexed by *OpenAI's* token ids, so your own 2000-token vocabulary cannot address them.

### A model you trained

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

Or via the Makefile, which fills in the paths from [Configuration](#configuration):

```bash
make run PROMPT="KING:" TOKENS=200
```

### A pretrained GPT-2

OpenAI's published weights, any of the four sizes, through this same transformer. Two stages: **set it up once**, then **prompt it**.

#### Setup

```bash
make setup                          # gpt2 small, ~500 MB
make gpt2-repl GPT2_DIR=models/gpt2
```

The verification step needs Python and PyTorch — see [Dependencies](#dependencies). Without them the model still downloads and runs; `setup` says that verification was skipped.

`make setup` downloads the checkpoint and tokenizer, confirms the file really is the model you asked for, dumps reference activations from PyTorch, verifies the forward pass against them, and prints the run commands. It is safe to re-run — anything already done is skipped, and an interrupted download resumes.

```bash
make setup MODEL_NAME=gpt2-medium
make setup MODEL_NAME=gpt2-xl SETUP_FLAGS=--int8
./scripts/setup_gpt2.sh gpt2-large --force      # or call it directly
```

| Size | Layers | Hidden | Heads | Params | Download |
| --- | --- | --- | --- | --- | --- |
| `gpt2` | 12 | 768 | 12 | 124M | ~500 MB |
| `gpt2-medium` | 24 | 1024 | 16 | 355M | 1.5 GB |
| `gpt2-large` | 36 | 1280 | 20 | 774M | ~3.1 GB |
| `gpt2-xl` | 48 | 1600 | 25 | 1.56B | ~6.4 GB |

The script runs on Linux, macOS and Windows (Git Bash / MSYS2 / WSL). It needs `curl` or `wget`, Go, and — for the verification step only — Python with the packages above. Without Python it still downloads and loads the model, and tells you verification was skipped.

#### Setup by hand

Useful when something goes wrong, or to understand what `make setup` is doing.

**1. Download.** Four files; three of them are small.

```bash
mkdir -p models/gpt2-large && cd models/gpt2-large
for f in model.safetensors config.json vocab.json merges.txt; do
  curl -LO https://huggingface.co/openai-community/gpt2-large/resolve/main/$f
done
cd ../..
```

`vocab.json` and `merges.txt` are **identical for all four sizes** — copy them from another model directory instead of downloading again. `config.json` is only needed by the Python reference dumper, not by the Go side.

**2. Check what you actually got.**

```bash
make inspect GPT2_DIR=models/gpt2-large
```

Prints the shape-defining tensors from the header. One look at `wte.weight [50257, 1280]` tells you which model this is — worth doing before spending time on it, and it doubles as a check that the download is not truncated.

**3. Load it.**

```bash
make load-gpt2 GPT2_DIR=models/gpt2-large
```

Reports params, layers, heads and context length. Proves the loader parses the file. Does **not** prove the numbers are right.

**4. Dump the reference and verify.** The only step that needs Python, once per model.

```bash
make gpt2-reference-local GPT2_DIR=models/gpt2-large
make verify-gpt2 GPT2_DIR=models/gpt2-large
```

`gpt2-reference-local` reads the weights already on disk rather than downloading a second copy, and uses fixed token ids so `vocab.json`/`merges.txt` are not required for it. Use `make gpt2-reference GPT2_MODEL_ID=gpt2-large` instead to let HuggingFace fetch its own copy.

A full pass looks like this — every tensor `ok`, ending at `logits`:

```
TENSOR         STATUS      MAX DIFF   MEAN DIFF     MAX REL    MEAN REL      SCALE
h.21.res1      ok         5.615e-03   7.434e-06    1.47e-06    1.22e-06     3823.9
ln_f           ok         1.633e-03   2.069e-06    5.51e-06    2.92e-06      296.5
logits         ok         8.469e-04   1.810e-04    6.23e-06    2.12e-06      135.8
```

`SCALE` is the largest magnitude in the reference tensor, and it is what makes the absolute columns readable: 5.6e-03 against a scale of 3823 is the eighth significant digit, not a bug. The mean relative error holding steady at ~1.2e-06 across all 24 layers is the real evidence the arithmetic is right.

**5. Generate.**

```bash
make gpt2-run GPT2_DIR=models/gpt2-large
```

#### Prompting it

Two commands. `gpt2-repl` is the one to use while exploring, since it loads the weights once.

```bash
go run -tags blas main.go gpt2-repl --dir models/gpt2
go run -tags blas main.go gpt2-run  --dir models/gpt2 --prompt "..." --tokens 60
```

Build once if you are iterating, and skip the recompile:

```bash
make build-blas
./gpt-go gpt2-repl --dir models/gpt2
```

Shared flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--dir` / `-d` | `models/gpt2` | Directory with `model.safetensors`, `vocab.json`, `merges.txt` |
| `--int8` | off | int8 weights — less memory, slower, ~1% logit error |
| `--heads` | `0` | Head count; 0 infers it from the hidden dim |

`gpt2-run` only:

| Flag | Default | Description |
| --- | --- | --- |
| `--prompt` / `-p` | `The area of Antarctica is` | Prompt |
| `--tokens` / `-n` | `40` | New tokens to generate |
| `--temperature` | `0.8` | 0 = greedy, deterministic, repetitive |
| `--top-k` / `-k` | `40` | Top-k pool; 0 = full distribution |
| `--rep-penalty` | `1.0` | 1.0 = off; ~1.15 fights loops |
| `--slide` | off | Keep going past the context limit |
| `--ignore-eos` | off | Keep generating past `<\|endoftext\|>` |
| `--show-tokens` | off | Per-token breakdown; `>` marks where the model took over |

Those are the flag defaults. The `make` targets pass their own values from [Configuration](#configuration), so `make gpt2-run` and a bare `gpt2-run` are not identical — the Makefile sends `TOKENS=50` and `REP_PENALTY=1.2`, for instance.

In the REPL, the same settings are slash commands — `/temp 0`, `/tokens 200`, `/topk`, `/reppen`, `/slide`, `/eos`, `/settings`, `/tokenize some text`, `/help`, `/exit`. Output streams as it is generated.

Use `--temperature 0` whenever you want to compare two runs; anything above it samples.

#### What to expect

```
The area of Antarctica is the largest continent on Earth, covering more than
half of the planet.

Antarctica was first sighted in 1869 by a British explorer, William Smith.
```

Fluent, confident and frequently wrong. That is a 355M model from 2019, not a loading bug — the verification step is what lets you tell those apart.

---

## Performance and memory

Measured on gpt2-medium (355M), 30 tokens, greedy, BLAS backend:

| | Peak memory | Speed |
| --- | --- | --- |
| Original (whole file in memory) | 3,874 MB | 22.9 tok/s |
| Streaming reads | 1,862 MB | 23.7 tok/s |
| Streaming + `--int8` | **1,086 MB** | 6.8 tok/s |

gpt2 small runs at ~76 tok/s and loads in well under a second. Sliding past the context window costs one prefill per 512 tokens — 1100 tokens generated through a 1024 context ran at 16.8 tok/s.

int8 is a genuine trade, not a free win: **~40% less memory, ~3.4× slower.** The accuracy cost, measured with the same verification harness, is ~2% mean relative error in the residual stream and **0.8% on the logits**, and it does not compound across layers. Greedy output stays coherent but does diverge, since 0.8% is enough to flip an argmax.

Worth it when memory is the binding constraint — gpt2-xl in particular — and not otherwise.

---

## Configuration

Every workflow has a Makefile target; `make help` lists them with their current values. Override any variable on the command line:

```bash
make gpt2-run GPT2_DIR=models/gpt2-medium GPT2_PROMPT="Antarctica is" TOKENS=80
make train EPOCHS=10 LR=0.0005
make verify-gpt2 GPT2_DIR=models/gpt2-large TOLERANCE=1e-2
make inspect FILE=models/gpt2-medium/reference_activations.safetensors
```

| Variable | Default | Used by |
| --- | --- | --- |
| `MODEL_NAME` | `gpt2` | `setup` — which size to download |
| `SETUP_FLAGS` | — | `setup` — `--force`, `--int8`, `--skip-verify` |
| `GPT2_DIR` | `models/gpt2` | every GPT-2 target; cascades to the weights and reference paths |
| `GPT2_PROMPT` | `The area of Antarctica is` | `gpt2-run`, `gpt2-reference*` |
| `GPT2_TOKENS` | `464,1989,286,30185,318` | `gpt2-reference-local` — ids for the prompt above |
| `TOLERANCE` | `1e-3` | `verify-gpt2` |
| `TOKENS`, `TEMPERATURE`, `TOP_K`, `REP_PENALTY` | `50`, `0.8`, `40`, `1.2` | `run`, `gpt2-run` |
| `INPUT`, `TOKENIZER`, `MODEL`, `VOCAB_SIZE`, `EPOCHS`, `LR`, `SEQ_LEN` | see `make help` | the training targets |

`GPT2_DIR` is the one that matters most — set it once and `load-gpt2`, `verify-gpt2`, `gpt2-run`, `gpt2-repl` and `inspect` all follow.

**Verification tolerances**, if you need to loosen them (for instance to measure int8 error):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tolerance` | `1e-3` | Absolute difference accepted outright, for tensors too small for a relative figure |
| `--rel-tolerance` | `1e-3` | Max relative error, worst element against the tensor's largest magnitude |
| `--mean-rel-tolerance` | `1e-4` | Mean relative error — the one that catches a wrong activation or a missing bias |
| `--show-all` | off | Print every tensor instead of stopping after the first failure |

---

## Testing

```bash
make test           # or: go test ./...
make test-verbose
go test -tags blas ./...
```

93 tests. The two worth knowing about, because they test things that otherwise fail silently:

**Fixture comparison.** The byte-level BPE is checked against 28 cases plus a round trip over all 256 byte values, generated from the real tokenizer. The fixtures are committed, so these pass without Python. Regenerate with `make gpt2-tokenizer-fixtures`.

**Numerical gradient checking.** `TestGradientsMatchNumericalGradient` perturbs 13 individual weights by ±0.01, measures the actual loss slope, and compares it against the analytic gradient — covering every gradient path including the bias terms. A gradient that is merely *computed* rather than *correct* passes every other kind of test; this one catches it.

---

## Dependencies

**Go:**

| Dependency               | When                   |
| ------------------------ | ---------------------- |
| `github.com/spf13/cobra` | always — CLI framework |
| `gonum.org/v1/gonum`     | only with `-tags blas` |

Default build has no external ML dependencies. gonum's BLAS is itself pure Go with Go assembly kernels, so even the `blas` build uses no cgo and cross-compiles normally.

**Python** — `scripts/requirements.txt`, and only for the two helper scripts:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements.txt
```

| Dependency     | Why                                                      |
| -------------- | -------------------------------------------------------- |
| `torch`        | the reference implementation we verify the forward pass against |
| `transformers` | loading GPT-2 to dump activations and tokenizer fixtures |
| `safetensors`  | the file format both sides exchange                      |

Nothing in Go calls Python. These run once per model, produce a file, and are then irrelevant — the CPU-only torch index matters because the default wheel pulls in several GB of CUDA for a single forward pass. The tokenizer fixtures are committed, so `go test ./...` passes on a machine with no Python at all.

---

## Credits

Now, I did not study machine learning and advanced math from scratch in a couple of days, that would be impossible. Therefore, I leveraged Gemini to help me understand the architecture and concepts, while I wrote the code. Claude was used at the end to find some critical bugs and flaws I introduced.

| Role                      | Who    |
| ------------------------- | ------ |
| Go Code                   | me     |
| Concept guidance & theory | Gemini |
| Bug fixes & improvements  | Claude |
