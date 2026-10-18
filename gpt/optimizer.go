package gpt

import "math"

// AdamW Optimizer with weight decay
// i've read that this is the gold standard optimizer used to train
// modern transformers like GPT. it tracks two moments for every single weight
// first moment -> exponential moving average of past gradients
// second moment -> exponential moving average of past suqared gradients

type AdamW struct {
	learningRate float32
	beta1        float32
	beta2        float32
	epsilon      float32
	weightDecay  float32
	timestep     int

	mBuffers map[*Matrix][]float32
	vBuffers map[*Matrix][]float32
}

func NewAdamW(learningRate, beta1, beta2, epsilon, weightDecay float32) *AdamW {
	return &AdamW{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		weightDecay:  weightDecay,
		timestep:     0,
		mBuffers:     make(map[*Matrix][]float32),
		vBuffers:     make(map[*Matrix][]float32),
	}
}

// step handles the matrix data and updates it using the accumulated gradients
func (opt *AdamW) Step(param *Matrix) {
	if param == nil || len(param.grad) == 0 {
		return
	}

	// buffer initialization
	if _, exists := opt.mBuffers[param]; !exists {
		opt.mBuffers[param] = make([]float32, len(param.data))
		opt.vBuffers[param] = make([]float32, len(param.data))
	}

	m := opt.mBuffers[param]
	v := opt.vBuffers[param]

	// bias correction handling
	biasCorrection1 := float32(1.0 - math.Pow(float64(opt.beta1), float64(opt.timestep)))
	biasCorrection2 := float32(1.0 - math.Pow(float64(opt.beta2), float64(opt.timestep)))

	lr := opt.learningRate
	for i := range param.data {
		g := param.grad[i]
		if g == 0 {
			continue
		}

		// weight decay
		if opt.weightDecay > 0 {
			param.data[i] -= lr * opt.weightDecay * param.data[i]
		}

		// update biased first moment
		m[i] = opt.beta1*m[i] + (1.0-opt.beta1)*g
		v[i] = opt.beta2*v[i] + (1.0-opt.beta2)*(g*g)

		// compute bias correctness
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2

		// parameter update
		param.data[i] -= lr * (mHat / (float32(math.Sqrt(float64(vHat))) + opt.epsilon))
	}
}
