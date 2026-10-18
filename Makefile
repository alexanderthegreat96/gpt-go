BINARY     := gpt-go
INPUT      := models/tiny_wiki/train.txt
TOKENIZER  := models/tiny_wiki/tokenizer_model_data.json
MODEL      := models/tiny_wiki/gpt_model.json
VOCAB_SIZE := 2000
EPOCHS     := 5
LR         := 0.001
SEQ_LEN    := 64
PRINT_EVERY := 50
PROMPT     := hello world
TOKENS     := 50
TEMPERATURE := 0.8
TOP_K      := 40
REP_PENALTY := 1.2
MODEL_NAME := gpt2
SETUP_FLAGS :=
GPT2_DIR   := models/gpt2
GPT2_WEIGHTS := $(GPT2_DIR)/model.safetensors
GPT2_REF   := $(GPT2_DIR)/reference_activations.safetensors
GPT2_MODEL_ID := gpt2
GPT2_PROMPT := The area of Antarctica is
# token ids for GPT2_PROMPT. every GPT-2 size shares one tokenizer, so these
# are valid for small, medium, large and xl alike
GPT2_TOKENS := 464,1989,286,30185,318
TOLERANCE  := 1e-3
FILE       := $(GPT2_WEIGHTS)

.SILENT:
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

.PHONY: all help build build-blas test test-verbose train-bpe train train-blas run clean gpt2-reference gpt2-reference-local gpt2-tokenizer-fixtures gpt2-run load-gpt2 verify-gpt2 inspect gpt2-run-int8 gpt2-repl gpt2-repl-int8 setup

all: build

help:
	printf '\n'
	printf 'Usage: make <target> [VAR=value ...]\n'
	printf '\n'
	printf 'Targets:\n'
	printf '  %-14s %s\n' 'help'         'show this message'
	printf '  %-14s %s\n' 'build'        'build $(BINARY) (pure Go backend)'
	printf '  %-14s %s\n' 'build-blas'   'build $(BINARY) with the gonum BLAS backend'
	printf '  %-14s %s\n' 'test'         'run all tests'
	printf '  %-14s %s\n' 'test-verbose' 'run all tests with verbose output'
	printf '  %-14s %s\n' 'train-bpe'    'train the BPE tokenizer on $(INPUT)'
	printf '  %-14s %s\n' 'train'        'train the GPT model (pure Go backend)'
	printf '  %-14s %s\n' 'train-blas'   'train the GPT model with the BLAS backend'
	printf '  %-14s %s\n' 'run'          'generate text from $(MODEL)'
	printf '  %-14s %s\n' 'setup'        'ONE COMMAND: download + check + verify a GPT-2, then say how to run it'
	printf '  %-14s %s\n' 'gpt2-reference' 'download GPT-2 + dump reference activations (needs python)'
	printf '  %-20s %s\n' 'gpt2-reference-local' 'same, but from weights already in $(GPT2_DIR)'
	printf '  %-14s %s\n' 'load-gpt2'    'load $(GPT2_WEIGHTS) and report what came out of it'
	printf '  %-14s %s\n' 'verify-gpt2'  'diff our forward pass against $(GPT2_REF)'
	printf '  %-14s %s\n' 'gpt2-run'     'generate text with the pretrained GPT-2 in $(GPT2_DIR)'
	printf '  %-20s %s\n' 'gpt2-run-int8' 'same, int8 weights -> ~40% less memory, ~3x slower'
	printf '  %-14s %s\n' 'gpt2-repl'    'interactive prompt loop, weights loaded once'
	printf '  %-20s %s\n' 'gpt2-repl-int8' 'same, with int8 weights'
	printf '  %-20s %s\n' 'gpt2-tokenizer-fixtures' 'regenerate the BPE test fixtures (needs python)'
	printf '  %-14s %s\n' 'inspect'      'dump the tensor shapes in FILE=<some.safetensors>'
	printf '  %-14s %s\n' 'clean'        'remove build artifacts'
	printf '\n'
	printf 'Variables (current values):\n'
	printf '  %-14s %s\n' 'BINARY'       '$(BINARY)'
	printf '  %-14s %s\n' 'INPUT'        '$(INPUT)'
	printf '  %-14s %s\n' 'TOKENIZER'    '$(TOKENIZER)'
	printf '  %-14s %s\n' 'MODEL'        '$(MODEL)'
	printf '  %-14s %s\n' 'VOCAB_SIZE'   '$(VOCAB_SIZE)'
	printf '  %-14s %s\n' 'EPOCHS'       '$(EPOCHS)'
	printf '  %-14s %s\n' 'LR'           '$(LR)'
	printf '  %-14s %s\n' 'SEQ_LEN'      '$(SEQ_LEN)'
	printf '  %-14s %s\n' 'PRINT_EVERY'  '$(PRINT_EVERY)'
	printf '  %-14s %s\n' 'PROMPT'       '$(PROMPT)'
	printf '  %-14s %s\n' 'TOKENS'       '$(TOKENS)'
	printf '  %-14s %s\n' 'TEMPERATURE'  '$(TEMPERATURE)'
	printf '  %-14s %s\n' 'TOP_K'        '$(TOP_K)'
	printf '  %-14s %s\n' 'REP_PENALTY'  '$(REP_PENALTY)'
	printf '  %-14s %s\n' 'GPT2_WEIGHTS' '$(GPT2_WEIGHTS)'
	printf '  %-14s %s\n' 'GPT2_REF'     '$(GPT2_REF)'
	printf '  %-14s %s\n' 'GPT2_MODEL_ID' '$(GPT2_MODEL_ID)'
	printf '  %-14s %s\n' 'TOLERANCE'    '$(TOLERANCE)'
	printf '\n'
	printf 'Examples:\n'
	printf '  make setup                              # gpt2 small, start here\n'
	printf '  make setup MODEL_NAME=gpt2-medium\n'
	printf '  make setup MODEL_NAME=gpt2-xl SETUP_FLAGS=--int8\n'
	printf '  make train EPOCHS=10 LR=0.0005\n'
	printf '  make train-blas EPOCHS=5\n'
	printf '  make run PROMPT="QUEEN:" TOKENS=120\n'
	printf '  make gpt2-reference GPT2_MODEL_ID=gpt2-medium\n'
	printf '  make gpt2-reference-local   # reuse weights already in $(GPT2_DIR)\n'
	printf '  make verify-gpt2\n'
	printf '  make inspect FILE=$(GPT2_REF)\n'
	printf '\n'

build:
	go build -o $(BINARY) .

build-blas:
	go build -tags blas -o $(BINARY) .

test:
	go test ./...

test-verbose:
	go test -v ./...

train-bpe:
	go run main.go train-bpe \
		--input $(INPUT) \
		--output $(TOKENIZER) \
		--vocab-size $(VOCAB_SIZE)

train:
	go run main.go train-gpt \
		--input $(INPUT) \
		--tokenizer $(TOKENIZER) \
		--output $(MODEL) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--seq-len $(SEQ_LEN) \
		--print-every $(PRINT_EVERY)

train-blas:
	go run -tags blas main.go train-gpt \
		--input $(INPUT) \
		--tokenizer $(TOKENIZER) \
		--output $(MODEL) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--seq-len $(SEQ_LEN) \
		--print-every $(PRINT_EVERY)

run:
	go run main.go run \
		--model $(MODEL) \
		--tokenizer $(TOKENIZER) \
		--prompt "$(PROMPT)" \
		--tokens $(TOKENS) \
		--temperature $(TEMPERATURE) \
		--top-k $(TOP_K) \
		--rep-penalty $(REP_PENALTY)

# the whole runbook in one go: download, check, verify, and print how to run
# it. MODEL_NAME picks the size, SETUP_FLAGS takes --force / --int8 /
# --skip-verify. safe to re-run, it skips whatever is already done
setup:
	bash scripts/setup_gpt2.sh $(MODEL_NAME) $(SETUP_FLAGS)

gpt2-reference:
	python scripts/dump_gpt2_reference.py \
		--model $(GPT2_MODEL_ID) \
		--prompt "$(GPT2_PROMPT)" \
		--out-dir $(GPT2_DIR)

# same thing, but reading the weights you already have on disk instead of
# downloading them again. needs config.json in $(GPT2_DIR) alongside
# model.safetensors, and uses explicit token ids so vocab.json/merges.txt
# are not required either
gpt2-reference-local:
	python scripts/dump_gpt2_reference.py \
		--model $(GPT2_DIR) \
		--tokens $(GPT2_TOKENS) \
		--prompt "$(GPT2_PROMPT)" \
		--out-dir $(GPT2_DIR)

load-gpt2:
	go run -tags blas main.go load-gpt2 \
		--weights $(GPT2_WEIGHTS)

verify-gpt2:
	go run -tags blas main.go verify-gpt2 \
		--weights $(GPT2_WEIGHTS) \
		--reference $(GPT2_REF) \
		--tolerance $(TOLERANCE)

gpt2-tokenizer-fixtures:
	python scripts/dump_gpt2_tokenizer_fixtures.py

gpt2-run:
	go run -tags blas main.go gpt2-run \
		--dir $(GPT2_DIR) \
		--prompt "$(GPT2_PROMPT)" \
		--tokens $(TOKENS) \
		--temperature $(TEMPERATURE) \
		--top-k $(TOP_K) \
		--rep-penalty $(REP_PENALTY)

gpt2-run-int8:
	go run -tags blas main.go gpt2-run \
		--dir $(GPT2_DIR) \
		--prompt "$(GPT2_PROMPT)" \
		--tokens $(TOKENS) \
		--temperature $(TEMPERATURE) \
		--top-k $(TOP_K) \
		--rep-penalty $(REP_PENALTY) \
		--int8

gpt2-repl:
	go run -tags blas main.go gpt2-repl --dir $(GPT2_DIR)

gpt2-repl-int8:
	go run -tags blas main.go gpt2-repl --dir $(GPT2_DIR) --int8

inspect:
	go run main.go inspect-safetensors --file $(FILE)

clean:
	go clean
	rm -f $(BINARY)
