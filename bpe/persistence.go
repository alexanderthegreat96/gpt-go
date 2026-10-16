package bpe

import (
	"encoding/json"
	"os"
)

type persistenceDTO struct {
	Data      map[int][]byte `json:"data"`
	MergeList []mergeRule    `json:"merge_list"`
}

func LoadTrainingData(filename string) ([]byte, error) {
	file, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	return file, nil
}

func SaveTokenizer(v *Vocabulary, filename string) error {
	dto := persistenceDTO{
		Data:      v.data,
		MergeList: v.mergeList,
	}

	fileData, err := json.Marshal(dto)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, fileData, 0644)
}

func LoadTokenizer(filename string) (*Vocabulary, error) {
	fileData, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var dto persistenceDTO

	if err := json.Unmarshal(fileData, &dto); err != nil {
		return nil, err
	}

	v := &Vocabulary{
		data:      dto.Data,
		merges:    make(map[[2]int]int),
		mergeList: dto.MergeList,
	}

	for _, rule := range v.mergeList {
		v.merges[[2]int{rule.Left, rule.Right}] = rule.NewId
	}

	return v, nil
}
