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

.PHONY: all build build-blas test clean train-bpe train run

all: build

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
