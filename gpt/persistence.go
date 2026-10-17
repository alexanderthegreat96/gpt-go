package gpt

import (
	"encoding/json"
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
}

func SaveModel(model *GPTModel, filename string) error {
	if model == nil {
		return os.ErrInvalid
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

		model.blocks[i] = &TransformerBlock{
			attention: &CausalAttention{
				wq:       wq,
				wk:       wk,
				wv:       wv,
				wo:       wo,
				numHeads: dto.NumHeads,
				headDim:  headDim,
				dModel:   dto.DModel,
			},
			ffn: &FeedForward{
				w1: w1,
				w2: w2,
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

	return model, nil
}
