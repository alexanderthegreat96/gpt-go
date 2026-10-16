BINARY     := gpt-go
INPUT      := train.txt
TOKENIZER  := tokenizer_model_data.json
MODEL      := gpt_model.json
VOCAB_SIZE := 2000
EPOCHS     := 5
LR         := 0.001
SEQ_LEN    := 64
PRINT_EVERY := 50
PROMPT     := KING:
TOKENS     := 50
TEMPERATURE := 0.8
TOP_K      := 40
REP_PENALTY := 1.2

.SILENT:
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

.PHONY: all help build build-blas test test-verbose train-bpe train train-blas run clean

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
	printf '\n'
	printf 'Examples:\n'
	printf '  make train EPOCHS=10 LR=0.0005\n'
	printf '  make train-blas EPOCHS=5\n'
	printf '  make run PROMPT="QUEEN:" TOKENS=120\n'
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

clean:
	go clean
	rm -f $(BINARY)
