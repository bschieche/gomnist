package mnist

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/bschieche/gomnist/utils/inputdata"
	"github.com/bschieche/gomnist/utils/slicefunc"
	"gonum.org/v1/gonum/mat"
)

// Hyperparameters is a struct of all parameters that are up to change or
// evaluation
type Hyperparameters struct {
	Lambda        float64
	NumHidden     int
	NumIters      int
	LearningRate  float64
	Momentum      float64
	EarlyStopping bool
	BatchSize     int
}

// CrossvalList is a struct of the same parameters as in hyperparameters, but
// with slices to give multiple values for cross validation
type CrossvalList struct {
	Lambda        []float64
	NumHidden     []int
	NumIters      []int
	LearningRate  []float64
	Momentum      []float64
	EarlyStopping []bool
	BatchSize     []int
}

var workers = runtime.NumCPU() * 4

// initialModel acts on a Hyperparameters instance, gets the input data of type *inputdata.Datas and
// returns a pointer on an initialized model object.
// Instead of random initialization, nonzero cosinus values are used in order to get comparable results
func (hp Hyperparameters) initialModel(datas *inputdata.Datas) *Model {

	// error handling
	ri, ci := datas.Training.Inputs.Dims()
	rt, ct := datas.Training.Targets.Dims()
	if ci != ct {
		panic("Number of training cases does not equal the number of targets.")
	}

	// create long vector of initial values for model parameters,
	// create new model object and transform initial values to matrices
	numParams := (ri + rt) * hp.NumHidden
	theta := mat.NewVecDense(numParams, nil)
	for i := 0; i < numParams; i++ {
		theta.SetVec(i, math.Cos(float64(i))*0.1)
	}
	model := &Model{
		Hyperparameters: hp,
		numParams:       numParams,
		numInput:        ri,
		numTarget:       rt,
		numCases:        ci,
		modelMat: &modelMat{
			inputToHidden: mat.NewDense(hp.NumHidden, ri, nil),
			hiddenToClass: mat.NewDense(rt, hp.NumHidden, nil),
		},
	}

	model.thetaToModel(theta)

	return model
}

// check acts on a Hyperparameter receiver and checks if they are valid
// and returns an error if one is not valid.
func (hp Hyperparameters) check(numCases int) error {
	if lambda := hp.Lambda; lambda < 0 {
		return fmt.Errorf("Lambda (%.4e) is supposed to be non-negative.\n", lambda)
	}
	if lr := hp.LearningRate; lr < 0 {
		return fmt.Errorf("Learning rate (%.4e) is supposed to be non-negative.\n", lr)
	}
	if mom := hp.Momentum; mom < 0 || mom >= 1 {
		return fmt.Errorf("Momentum (%.4e) is supposed to be between 0 and 1.\n", mom)
	}
	if n := hp.NumHidden; n < 0 || (n == 0 && hp.NumIters > 0) {
		return fmt.Errorf("Number of hidden units (%d) is supposed to be at least 1 or 0 when n_iters<1.\n", n)
	}
	if n := hp.BatchSize; (n < 1 || n > numCases) && hp.NumIters > 0 {
		return fmt.Errorf("Size of mini batches (%d) is supposed to be at least 1 and smaller than the number of training cases %d.\n", n, numCases)
	}

	return nil
}

// RunAll runs the optimization for given hyperparameters and data and
// returns a map with string keys (for training, validation, and test) and
// values of type accuracy
func (hp Hyperparameters) RunAll(datas *inputdata.Datas, print bool) (*Model, error) {
	// check if hp is a valid hyperparameter set
	_, numCases := datas.Training.Inputs.Dims()
	err := hp.check(numCases)
	if err != nil {
		return nil, err
	}

	// initialize model
	model := hp.initialModel(datas)
	fmt.Println("Model is initialized.")

	// test gradient
	if model.NumIters != 0 {
		model.testGradient(datas.Training, model.Lambda)
	}
	// optimize
	model.optimize(datas, print)
	fmt.Println("Optimization finished.")

	// report loss
	model.getLoss(datas, print)
	return model, nil
}

// CrossVal gets the data and a variable of type crossvalList as its receiver,
// runs all possible combinations sequentially and returns the best
// hyperparameters and the corresponding loss.
// see concurrentCrossVal for a concurrent version
func (hps CrossvalList) CrossVal(datas *inputdata.Datas) (*Model, error) {

	bestLoss := math.MaxFloat64
	bestModel := &Model{}

	// get all possible parameter combinations into a hyperparameter
	// slice and run the model
	allHp := hps.getParCombis()
	start := time.Now()
	for _, hp := range allHp {
		fmt.Println(hp)
		model, err := hp.RunAll(datas, true)
		if err != nil {
			fmt.Println(err, "Set of hyperparameters is not used. Continue with next.")
			continue
		}

		// update best loss and best hyperparameter set
		if l := model.FinalLossMap["validation"].LossWithoutLambda; l < bestLoss {
			bestLoss = l
			bestModel = model.copy()
		}
	}

	// report ellapsed time
	finish := time.Now()
	fmt.Printf("Sequential cross validation took %v time.\n", finish.Sub(start))

	// report whether no valid set was given at all or print the
	// results for the best model
	if bestModel.FinalLossMap["training"].Loss == 0 {
		return nil, fmt.Errorf("No valid parameter set in the whole list given.\n")
	}
	fmt.Printf("\nCross validation: The best loss %.4e is obtainded for hyperparameters:\n%v\n",
		bestLoss, bestModel.Hyperparameters)
	fmt.Printf("The corresponding error rate is %.4f.\n",
		bestModel.FinalLossMap["validation"].ErrorRate)
	return bestModel, nil
}

// ConcurrentCrossVal gets the data and a variable of type CrossvalList
// as its receiver, runs all possible combinations sequentially and
// returns the best hyperparameters and the corresponding loss
// see CrossVal for a sequential version
func (hps CrossvalList) ConcurrentCrossVal(datas *inputdata.Datas) (*Model, error) {

	// initialize best model
	finalLossMap := make(map[string]accuracy)
	finalLossMap["validation"] = accuracy{LossWithoutLambda: math.MaxFloat64}
	bestModel := &safemodel{
		Model: &Model{FinalLossMap: finalLossMap},
	}

	// get all possible parameter combinations into a hyperparameter slice
	allHp := hps.getParCombis()

	// channel for the resulting models
	results := make(chan *Model, len(allHp))

	// run all models and limit in-progress jobs to #workers
	sem := make(chan struct{}, workers)
	start := time.Now()
	counter := 0
	for _, hp := range allHp {
		go func(hp Hyperparameters) {
			sem <- struct{}{}
			counter++
			fmt.Println(counter)
			hp.doJob(results, datas)
			<-sem
		}(hp)
	}

	// collect the results and store the best so far obtained
	for range allHp {
		model := <-results
		model.checkIfBest(bestModel)
	}

	// report ellapsed time
	finish := time.Now()
	fmt.Printf("Concurrent cross validation took %v time.\n", finish.Sub(start))

	// report whether no valid set was given at all or print the
	// results for the best model
	if bestModel.FinalLossMap["training"].ErrorRate == 0 {
		return nil, fmt.Errorf("No valid parameter set in the whole list given.\n")
	}
	fmt.Printf("\nCross validation: The best loss %.4e is obtainded for hyperparameters:\n%v\n",
		bestModel.FinalLossMap["validation"].LossWithoutLambda, bestModel.Hyperparameters)
	fmt.Printf("The corresponding error rate is %.4f.\n",
		bestModel.FinalLossMap["validation"].ErrorRate)
	return bestModel.Model, nil
}

// doJob gets a set of hyperparameters, the results channel and the data.
// runs the model and sends it into the results channel.
func (hp Hyperparameters) doJob(results chan<- *Model, datas *inputdata.Datas) {
	model, err := hp.RunAll(datas, false)
	if err != nil {
		fmt.Println(err, "Set of hyperparameters is not used. Continue with next.")
	}
	results <- model
}

// getParCombis acts on a receiver of type CrossvalList and returns a slice of Hyperparameters
// of all possible configurations
func (hps CrossvalList) getParCombis() []Hyperparameters {
	// get the number of values for each hyperparameter
	// and calculate the number of combinations
	lengths := hps.lengths()
	numCombis := slicefunc.IntSliceProd(lengths)

	// get all combinations in terms of indices
	// Alternative: use package gonum/stat/combin, Cartesian
	hpCombis := make([][]int, numCombis)
	slicefunc.GetCombis(hpCombis, lengths, 0, numCombis)

	// make a slice of Hyperparameters by looping over all
	// combinations (given as indices) and extracting the corresponding
	// values from the cross validation list
	allHp := make([]Hyperparameters, len(hpCombis))
	for i, hpCombi := range hpCombis {
		allHp[i] = Hyperparameters{
			Lambda:        hps.Lambda[hpCombi[0]],
			NumHidden:     hps.NumHidden[hpCombi[1]],
			NumIters:      hps.NumIters[hpCombi[2]],
			LearningRate:  hps.LearningRate[hpCombi[3]],
			Momentum:      hps.Momentum[hpCombi[4]],
			EarlyStopping: hps.EarlyStopping[hpCombi[5]],
			BatchSize:     hps.BatchSize[hpCombi[6]],
		}
	}

	return allHp
}

// lengths acts on a receiver of type CrossvalList and returns an int slice of
// the number of values for each hyperparameter given
func (hps CrossvalList) lengths() []int {
	l := make([]int, 7)
	l[0] = len(hps.Lambda)
	l[1] = len(hps.NumHidden)
	l[2] = len(hps.NumIters)
	l[3] = len(hps.LearningRate)
	l[4] = len(hps.Momentum)
	l[5] = len(hps.EarlyStopping)
	l[6] = len(hps.BatchSize)
	return l
}

// hp fulfills Stringer interface
func (hp Hyperparameters) String() string {
	s := fmt.Sprintf("lambda: %23.4e\nlearning rate: %16.4e\nmomentum multiplier: %.4e\nnumber of hidden units: %7d\nnumber of iterations: %9d\nmini batch size: %14d\nearly stopping activated: %t\n", hp.Lambda, hp.LearningRate, hp.Momentum, hp.NumHidden, hp.NumIters, hp.BatchSize, hp.EarlyStopping)
	return s
}
