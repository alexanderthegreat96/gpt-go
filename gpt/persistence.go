package gpt

import (
	"encoding/json"
	"fmt"
	"os"
)

type MatrixDTO struct {
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
	Data []float32 `json:"data"`
}

func LoadTrainingData(filename string) ([]byte, error) {
	file, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return file, nil
}

func serializeMatrix(m *Matrix) *MatrixDTO {
	if m == nil {
		return nil
	}
	dataCopy := make([]float32, len(m.data))
	copy(dataCopy, m.data)

	return &MatrixDTO{
		Rows: m.rows,
		Cols: m.cols,
		Data: dataCopy,
	}
}

func deserializeMatrix(dto *MatrixDTO) (*Matrix, error) {
	if dto == nil {
		return nil, nil
	}
	return NewMatrixFromData(dto.Rows, dto.Cols, dto.Data)
}

type TransformerBlockDTO struct {
	Wq       *MatrixDTO `json:"wq"`
	Wk       *MatrixDTO `json:"wk"`
	Wv       *MatrixDTO `json:"wv"`
	Wo       *MatrixDTO `json:"wo"`
	W1       *MatrixDTO `json:"w1"`
	W2       *MatrixDTO `json:"w2"`
	Ln1Gamma *MatrixDTO `json:"ln1_gamma"`
	Ln1Beta  *MatrixDTO `json:"ln1_beta"`
	Ln2Gamma *MatrixDTO `json:"ln2_gamma"`
	Ln2Beta  *MatrixDTO `json:"ln2_beta"`
	Epsilon  float32    `json:"epsilon"`

	// everything below only exists on models that came out of a pretrained
	// checkpoint, hence omitempty -> a model we trained ourselves writes the
	// exact same json it always did, and old model files still load as before.
	//
	// these HAVE to be here even though we would rather load gpt-2 from
	// safetensors every time. if they were missing, saving a loaded gpt-2 and
	// reading it back would quietly drop its biases and flip its activation
	// function, which is precisely the kind of silent wrongness this whole
	// exercise is built to avoid
	Bq          *MatrixDTO `json:"bq,omitempty"`
	Bk          *MatrixDTO `json:"bk,omitempty"`
	Bv          *MatrixDTO `json:"bv,omitempty"`
	Bo          *MatrixDTO `json:"bo,omitempty"`
	B1          *MatrixDTO `json:"b1,omitempty"`
	B2          *MatrixDTO `json:"b2,omitempty"`
	UseTanhGELU bool       `json:"use_tanh_gelu,omitempty"`
}

type GPTModelDTO struct {
	VocabSize    int                   `json:"vocab_size"`
	DModel       int                   `json:"d_model"`
	NumLayers    int                   `json:"num_layers"`
	NumHeads     int                   `json:"num_heads"`
	MaxSeqLen    int                   `json:"max_seq_len"`
	Epsilon      float32               `json:"epsilon"`
	TokenEmbed   *MatrixDTO            `json:"token_embed"`
	PosEmbed     *MatrixDTO            `json:"pos_embed"`
	Blocks       []TransformerBlockDTO `json:"blocks"`
	FinalLnGamma *MatrixDTO            `json:"final_ln_gamma"`
	FinalLnBeta  *MatrixDTO            `json:"final_ln_beta"`
	LmHead       *MatrixDTO            `json:"lm_head"`

	// whether the lm head above is really just the token embeddings seen from
	// the other side. only affects the parameter count, but losing it would
	// make a reloaded gpt-2 claim 163M parameters instead of 124M
	LmHeadTied bool `json:"lm_head_tied,omitempty"`
}

func SaveModel(model *GPTModel, filename string) error {
	if model == nil {
		return os.ErrInvalid
	}

	// a quantized model has no float32 weights left, so serializing it would
	// write a file full of nulls that loads without complaint and produces
	// garbage. refuse instead
	for i, block := range model.blocks {
		for name, weight := range map[string]*Matrix{
			"wq": block.attention.wq, "wk": block.attention.wk,
			"wv": block.attention.wv, "wo": block.attention.wo,
			"w1": block.ffn.w1, "w2": block.ffn.w2,
		} {
			if weight != nil && weight.IsQuantized() {
				return fmt.Errorf("SaveModel: block %d %s is quantized -> our json format only stores float32 weights", i, name)
			}
		}
	}
	if model.lmHead != nil && model.lmHead.IsQuantized() {
		return fmt.Errorf("SaveModel: the lm head is quantized -> our json format only stores float32 weights")
	}

	blocksDTO := make([]TransformerBlockDTO, len(model.blocks))
	for i, b := range model.blocks {
		blocksDTO[i] = TransformerBlockDTO{
			Wq:       serializeMatrix(b.attention.wq),
			Wk:       serializeMatrix(b.attention.wk),
			Wv:       serializeMatrix(b.attention.wv),
			Wo:       serializeMatrix(b.attention.wo),
			W1:       serializeMatrix(b.ffn.w1),
			W2:       serializeMatrix(b.ffn.w2),
			Ln1Gamma: serializeMatrix(b.ln1Gamma),
			Ln1Beta:  serializeMatrix(b.ln1Beta),
			Ln2Gamma: serializeMatrix(b.ln2Gamma),
			Ln2Beta:  serializeMatrix(b.ln2Beta),
			Epsilon:  b.epsilon,

			// nil in, nil out -> omitempty keeps these out of the file
			// entirely for models that never had them
			Bq:          serializeMatrix(b.attention.bq),
			Bk:          serializeMatrix(b.attention.bk),
			Bv:          serializeMatrix(b.attention.bv),
			Bo:          serializeMatrix(b.attention.bo),
			B1:          serializeMatrix(b.ffn.b1),
			B2:          serializeMatrix(b.ffn.b2),
			UseTanhGELU: b.ffn.useTanhGELU,
		}
	}

	dto := GPTModelDTO{
		VocabSize:    model.vocabSize,
		DModel:       model.dModel,
		NumLayers:    model.numLayers,
		NumHeads:     model.numHeads,
		MaxSeqLen:    model.maxSeqLen,
		Epsilon:      model.epsilon,
		TokenEmbed:   serializeMatrix(model.tokenEmbed.weights),
		PosEmbed:     serializeMatrix(model.positionEmbed.weights),
		Blocks:       blocksDTO,
		FinalLnGamma: serializeMatrix(model.finalLnGamma),
		FinalLnBeta:  serializeMatrix(model.finalLnBeta),
		LmHead:       serializeMatrix(model.lmHead),
		LmHeadTied:   model.lmHeadTied,
	}

	fileData, err := json.Marshal(dto)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, fileData, 0644)
}

func LoadModel(filename string) (*GPTModel, error) {
	fileData, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var dto GPTModelDTO
	if err := json.Unmarshal(fileData, &dto); err != nil {
		return nil, err
	}

	tokenEmbedWeights, err := deserializeMatrix(dto.TokenEmbed)
	if err != nil {
		return nil, err
	}

	posEmbedWeights, err := deserializeMatrix(dto.PosEmbed)
	if err != nil {
		return nil, err
	}

	maxSeqLen := dto.MaxSeqLen
	if maxSeqLen == 0 {
		maxSeqLen = posEmbedWeights.rows
	}

	model := &GPTModel{
		vocabSize: dto.VocabSize,
		dModel:    dto.DModel,
		numLayers: dto.NumLayers,
		numHeads:  dto.NumHeads,
		maxSeqLen: maxSeqLen,
		epsilon:   dto.Epsilon,
		blocks:    make([]*TransformerBlock, dto.NumLayers),
	}

	model.tokenEmbed = &Embedding{
		vocabSize: dto.VocabSize,
		hiddenDim: dto.DModel,
		weights:   tokenEmbedWeights,
	}

	model.positionEmbed = &PositionalEmbedding{
		maxSeqLen: maxSeqLen,
		hiddenDim: dto.DModel,
		weights:   posEmbedWeights,
	}

	headDim := dto.DModel / dto.NumHeads
	for i, bDto := range dto.Blocks {
		wq, _ := deserializeMatrix(bDto.Wq)
		wk, _ := deserializeMatrix(bDto.Wk)
		wv, _ := deserializeMatrix(bDto.Wv)
		wo, _ := deserializeMatrix(bDto.Wo)
		w1, _ := deserializeMatrix(bDto.W1)
		w2, _ := deserializeMatrix(bDto.W2)
		ln1Gamma, _ := deserializeMatrix(bDto.Ln1Gamma)
		ln1Beta, _ := deserializeMatrix(bDto.Ln1Beta)
		ln2Gamma, _ := deserializeMatrix(bDto.Ln2Gamma)
		ln2Beta, _ := deserializeMatrix(bDto.Ln2Beta)

		// the optional pretrained extras. deserializeMatrix hands back nil for
		// a nil DTO, so an older model file lands here with no biases at all
		// and the erf GELU, exactly as it behaved before any of this existed
		bq, _ := deserializeMatrix(bDto.Bq)
		bk, _ := deserializeMatrix(bDto.Bk)
		bv, _ := deserializeMatrix(bDto.Bv)
		bo, _ := deserializeMatrix(bDto.Bo)
		b1, _ := deserializeMatrix(bDto.B1)
		b2, _ := deserializeMatrix(bDto.B2)

		dInner := 0
		if w1 != nil {
			dInner = w1.cols
		}

		model.blocks[i] = &TransformerBlock{
			attention: &CausalAttention{
				wq:       wq,
				wk:       wk,
				wv:       wv,
				wo:       wo,
				numHeads: dto.NumHeads,
				headDim:  headDim,
				dModel:   dto.DModel,
				bq:       bq,
				bk:       bk,
				bv:       bv,
				bo:       bo,
			},
			ffn: &FeedForward{
				dModel:      dto.DModel,
				dInner:      dInner,
				w1:          w1,
				w2:          w2,
				b1:          b1,
				b2:          b2,
				useTanhGELU: bDto.UseTanhGELU,
			},
			ln1Gamma: ln1Gamma,
			ln1Beta:  ln1Beta,
			ln2Gamma: ln2Gamma,
			ln2Beta:  ln2Beta,
			epsilon:  bDto.Epsilon,
		}
	}

	model.finalLnGamma, _ = deserializeMatrix(dto.FinalLnGamma)
	model.finalLnBeta, _ = deserializeMatrix(dto.FinalLnBeta)
	model.lmHead, _ = deserializeMatrix(dto.LmHead)
	model.lmHeadTied = dto.LmHeadTied

	return model, nil
}
