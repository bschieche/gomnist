// Package mnist is able to set up a neural network with one hidden layer
// and a softmax as output.
// data needs to be of type inputdata.Datas.
// Each column represents a data case (i.e. an image in this specific case).
// The package was especially created for the MNIST data set to classify handwritten
// digits, but is not restricted to this application.
package mnist

import (
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sync"

	"github.com/bschieche/gomnist/utils/inputdata"
	"github.com/bschieche/gomnist/utils/matfunc"
	"github.com/bschieche/gomnist/utils/slicefunc"
	"gonum.org/v1/gonum/mat"
)

// Model contains all information of a model as well
// as the results (model parameters and losses) after
// the optimization.
type Model struct {
	Hyperparameters
	numParams int
	numInput  int
	numTarget int
	numCases  int
	*modelMat
	FinalLossMap map[string]accuracy
	lossPerIter
}

// safemodel contains a model and a mutex in order to
// store and access a model in concurrent cross validation.
type safemodel struct {
	*Model
	sync.Mutex
}

// model_mat contains the model parameters or their
// derivatives split up into 2 matrices:
// 1. transformation from input units to hidden units
// 2. transformation from hidden units to output units
type modelMat struct {
	inputToHidden *mat.Dense // numHidden x numInput (256)
	hiddenToClass *mat.Dense // numTarget (10=softmax) x numHidden
}

// early_stopping stores all relevant information of the best
// so far obtained classification loss during optimization
type earlyStopping struct {
	theta          *mat.VecDense
	validationLoss float64
	afterNumIters  int
}

// accuracy collects the loss with and without weight decay and
// the classification error rate
type accuracy struct {
	ErrorRate         float64
	Loss              float64
	LossWithoutLambda float64
}

// lossPerIter is a struct of slices especially useful for plotting
// the learning curve. It contains all iteration steps and the
// corresponding losses for training and validation data
type lossPerIter struct {
	iters            []float64
	trainingLosses   []float64
	validationLosses []float64
}

// propagation collects the matrices calculated with forward
// propagation and used for back propagation.
type propagation struct {
	hiddenInput  *mat.Dense
	hiddenOutput *mat.Dense
	classInput   *mat.Dense
}

// MisclassImg holds the information of misclassified images
// for a learned model with respect to the test data
type MisclassImg struct {
	Predictions []int
	Targets     []int
}

// optimize acts on a model receiver and gets all data and a flag
// if output information should be printed and plotted.
// It performs Stochastic Gradient descent with our without
// momentum. As regularization techniques weight decay and early
// stopping can be chosen
func (model *Model) optimize(datas *inputdata.Datas, printAndPlot bool) {

	// initialize model parameters, momentum speed, loss slices, and early stopping opject
	theta := model.modelToTheta()
	trainingLosses := make([]float64, 0, model.NumIters)
	validationLosses := make([]float64, 0, model.NumIters)
	iters := make([]float64, 0, model.NumIters)
	momSpeed := mat.NewVecDense(model.numParams, nil)
	bestSoFar := &earlyStopping{}
	if model.EarlyStopping {
		bestSoFar.theta = mat.NewVecDense(model.numParams, nil)
		bestSoFar.theta.CopyVec(theta)
		bestSoFar.validationLoss = math.MaxFloat64
		bestSoFar.afterNumIters = -1
	}

	// optimization
	for i := 0; i < model.NumIters; i++ {
		// selection of next mini batch
		mbStart := (i * model.BatchSize) % (model.numCases)
		mbData := &inputdata.Data{
			Inputs: matfunc.SliceCont(datas.Training.Inputs,
				0, model.numInput, mbStart, mbStart+model.BatchSize),
			Targets: matfunc.SliceCont(datas.Training.Targets,
				0, model.numTarget, mbStart, mbStart+model.BatchSize),
		}

		// get the gradient
		gradient := model.dLoss(mbData, model.Lambda).modelToTheta()

		// update model parameters
		momSpeed.ScaleVec(model.Momentum, momSpeed)
		momSpeed.SubVec(momSpeed, gradient)
		theta.AddScaledVec(theta, model.LearningRate, momSpeed)
		model.thetaToModel(theta)

		// get current losses for training and validation data
		trainingLoss := model.loss(datas.Training, 0, nil)
		validationLoss := model.loss(datas.Validation, 0, nil)
		iters = append(iters, float64(i+1))
		trainingLosses = append(trainingLosses, trainingLoss)
		validationLosses = append(validationLosses, validationLoss)

		// update early stopping object in case a new best so far loss is reached
		if model.EarlyStopping && validationLoss < bestSoFar.validationLoss {
			bestSoFar.theta.CopyVec(theta)
			bestSoFar.validationLoss = validationLoss
			bestSoFar.afterNumIters = i + 1
		}

		// 10 intermediate loss outputs
		if printAndPlot && (i+1)%int(math.Floor(float64(model.NumIters)/10.0+0.5)) == 0 {
			fmt.Printf("After %d optimization iterations training data loss is %f, and validation data loss is %f\n", i+1, trainingLoss, validationLoss)
		}
	}

	// save losses in the model for plotting
	model.lossPerIter = lossPerIter{iters, trainingLosses, validationLosses}

	// gradient check
	if model.NumIters != 0 {
		model.testGradient(datas.Training, model.Lambda)
	}

	// early stopping
	if model.EarlyStopping {
		//reporting
		if printAndPlot {
			fmt.Printf("Early stopping: validation loss was lowest after %d iterations.\n",
				bestSoFar.afterNumIters)
		}

		// restore saved model parameters in matrix form and the number of used iterations
		model.thetaToModel(bestSoFar.theta)
		model.NumIters = bestSoFar.afterNumIters
	}

	// plotting
	if printAndPlot {
		fh, err := os.OpenFile("learning_curve.png", os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Printf("No plot generated. Error: %s\n", err)
			return
		}
		defer fh.Close()
		err = plotLearningCurve(fh, model.lossPerIter)
		if err != nil {
			fmt.Printf("No plot generated. Error: %s\n", err)
		}
	}
}

// loss acts on a model receiver and gets data of type inputdata.Data, Lambda
// and a map with indices as keys and floats as values. In case off is nil
// the normal loss of the model is calculated. Otherwise the model parameters
// are shifted at the given index positions with respect to the corresponding
// values. This functionality is especially implemented for gradient checking
// by means of finite difference formulas. The function returns the loss as
// one single float.
func (model *Model) loss(data *inputdata.Data, lambda float64, off map[int]float64) float64 {

	// model parameters to use
	theta := model.modelToTheta()
	modelInUse := model.copy()

	// shift model parameters in case off contains indices
	if off != nil {
		for key, value := range off {
			theta.SetVec(key, theta.AtVec(key)+value)
		}
		modelInUse.thetaToModel(theta)
	}

	// error handling
	numInput, numCases := data.Inputs.Dims()
	if numInput != model.numInput {
		log.Fatal("Rows of input data does not match number of input units.\n")
	}

	// forward propagation
	prop := modelInUse.predict(data, numCases)

	// log propability
	logClassProb := matfunc.LogProb(prop.classInput)

	// loss calculation
	classificationLoss := -matfunc.SumElem(logClassProb, data.Targets) / float64(numCases)
	if lambda == 0 {
		return classificationLoss
	}

	//weight decay loss
	wdLoss := matfunc.SquareSum(theta) / 2 * lambda

	return classificationLoss + wdLoss
}

// dLoss acts on a model receiver and gets data of type inputdata.Data and lambda.
// It performs forward and backward propagation and returns a model_mat object
// which contains the derivative of the loss with respect to all model parameters.
func (model *Model) dLoss(data *inputdata.Data, lambda float64) *modelMat {

	// error handling
	numInput, _ := data.Inputs.Dims()
	if numInput != model.numInput {
		log.Fatal("Number of rows of input data does not match number of input units.\n")
	}
	numTarget, numCases := data.Targets.Dims()
	if numTarget != model.numTarget {
		log.Fatal("Number of rows of target data does not match number of output units.\n")
	}

	// forward propagation
	prop := model.predict(data, numCases)

	// softmax: numTarget x numCases
	logClassProb := matfunc.LogProb(prop.classInput)
	classProb := matfunc.ExpElem(logClassProb)

	// initialize backward propagation matrices
	W1 := mat.NewDense(model.NumHidden, numInput, nil)
	W2 := mat.NewDense(numTarget, model.NumHidden, nil)
	delta3 := mat.NewVecDense(model.numTarget, nil)
	delta2 := mat.NewDense(model.NumHidden, 1, nil)
	helpMat := mat.NewDense(model.numTarget, model.NumHidden, nil)

	// back propagation for each data case
	for j := 0; j < numCases; j++ {
		// n_target x 1
		delta3.SubVec(classProb.ColView(j), data.Targets.ColView(j))

		// (numHidden x numTarget) * (numTarget x 1) = numHidden x 1
		delta2.Mul(model.hiddenToClass.T(), delta3)
		delta2.MulElem(delta2, matfunc.LogisticGradMat(prop.hiddenInput.ColView(j)))

		// (numTarget x 1) * (1 x numHidden) = numTarget x numHidden
		helpMat.Mul(delta3, prop.hiddenOutput.ColView(j).T())
		W2.Add(W2, helpMat)
		helpMat.Reset()

		// (numHidden x 1) * (1 x numInput) = numHidden x numInput
		helpMat.Mul(delta2, data.Inputs.ColView(j).T())
		W1.Add(W1, helpMat)
		helpMat.Reset()
	}

	// gradient without weight decay
	W1.Scale(1.0/float64(numCases), W1)
	W2.Scale(1.0/float64(numCases), W2)

	// gradient of weight decay loss term
	inputToHiddenGrad := mat.DenseCopyOf(model.inputToHidden)
	inputToHiddenGrad.Scale(lambda, inputToHiddenGrad)
	hiddenToClassGrad := mat.DenseCopyOf(model.hiddenToClass)
	hiddenToClassGrad.Scale(lambda, hiddenToClassGrad)

	// overall gradient matrices
	inputToHiddenGrad.Add(inputToHiddenGrad, W1)
	hiddenToClassGrad.Add(hiddenToClassGrad, W2)

	return &modelMat{inputToHiddenGrad, hiddenToClassGrad}
}

// predict acts on a model receiver and gets data of type inputdata.Data
// and the number of data cases and performs the forward propagation
// up to the input to the softmax. It returns all matrices in a propagation
// struct.
func (model *Model) predict(data *inputdata.Data, numCases int) *propagation {
	prop := &propagation{}

	// input to the hidden units
	prop.hiddenInput = mat.NewDense(model.NumHidden, numCases, nil)
	prop.hiddenInput.Mul(model.inputToHidden, data.Inputs) // numHidden x numCases
	// output of the hidden units
	prop.hiddenOutput = matfunc.LogisticMat(prop.hiddenInput) // numHidden x numCases

	// input to the components of the softmax.
	prop.classInput = mat.NewDense(model.numTarget, numCases, nil)
	prop.classInput.Mul(model.hiddenToClass, prop.hiddenOutput) // numTarget x numCases

	return prop
}

// thetaToModel acts on a model receiver. It gets the model parameters (or gradient parameters)
// in the form of one long vector and restores it as matrices of type model_mat in the receiver model.
func (model *Model) thetaToModel(theta *mat.VecDense) {
	// error handling
	if model.numParams != theta.Len() {
		panic(mat.ErrShape)
	}

	// parameters as slice
	thetaSlice := theta.RawVector().Data

	// fetch dimensions
	numInput := model.numInput
	numTarget := model.numTarget
	numHidden := model.NumHidden

	// identify breaking point and copy data into the modelMat matrices
	id := numInput * numHidden
	model.inputToHidden.Copy(mat.NewDense(numHidden, numInput, thetaSlice[:id]))
	model.hiddenToClass.Copy(mat.NewDense(numTarget, numHidden, thetaSlice[id:]))
}

// modelToTheta acts on a model receiver. It returns a long vector
// of all model parameters.
func (model *modelMat) modelToTheta() *mat.VecDense {
	// return vector of length 0 in case matrices are zero dimensional
	numHidden, numInput := model.inputToHidden.Dims()
	if numHidden == 0 || numInput == 0 {
		return mat.NewVecDense(0, nil)
	}

	// append all backing slice data
	parameters := append(model.inputToHidden.RawMatrix().Data,
		model.hiddenToClass.RawMatrix().Data...)

	return mat.NewVecDense(len(parameters), parameters)
}

// copy copies a variable of type model into a new variable.
func (model *Model) copy() *Model {

	finalLossMap := make(map[string]accuracy, 3)
	finalLossMap["training"] = model.FinalLossMap["training"]
	finalLossMap["test"] = model.FinalLossMap["test"]
	finalLossMap["validation"] = model.FinalLossMap["validation"]

	numIters := len(model.lossPerIter.iters)
	iters := make([]float64, numIters)
	copy(iters, model.lossPerIter.iters)

	numTL := len(model.lossPerIter.trainingLosses)
	trainingLosses := make([]float64, numTL)
	copy(trainingLosses, model.lossPerIter.trainingLosses)

	numVL := len(model.lossPerIter.validationLosses)
	validationLosses := make([]float64, numVL)
	copy(validationLosses, model.lossPerIter.validationLosses)

	if numVL != numIters || numTL != numIters {
		log.Println("Warning: slices in lossPerIter should have the same length!")
	}

	return &Model{
		Hyperparameters: model.Hyperparameters,
		numParams:       model.numParams,
		numInput:        model.numInput,
		numTarget:       model.numTarget,
		numCases:        model.numCases,
		modelMat:        model.modelMat.copy(),
		FinalLossMap:    finalLossMap,
		lossPerIter: lossPerIter{
			iters:            iters,
			trainingLosses:   trainingLosses,
			validationLosses: validationLosses,
		},
	}
}

// copy copies a variable of type modelMat into a new variable.
func (model *modelMat) copy() *modelMat {
	modelCopy := &modelMat{}
	modelCopy.inputToHidden = mat.NewDense(0, 0, nil)
	modelCopy.inputToHidden.Clone(model.inputToHidden)
	modelCopy.hiddenToClass = mat.NewDense(0, 0, nil)
	modelCopy.hiddenToClass.Clone(model.hiddenToClass)

	return modelCopy
}

// testGradient acts on a model receiver and gets the data as input
// as well as lambda, which is used instead of model.lambda.
// It compares the analytic error (dLoss function) with numerical
// difference and reports if they differ more than a constant threshold
// of 1e-5. The program doesn't stop.
func (model *Model) testGradient(data *inputdata.Data, lambda float64) {
	const (
		h   = 1e-2 // step size for finite difference formula
		tol = 1e-5 // tolerance for gradient test
	)
	// coefficients for finite difference formula
	numDiff := map[float64]float64{
		-4: 1.0 / 280.0,
		-3: -4.0 / 105.0,
		-2: 1.0 / 5.0,
		-1: -4.0 / 5.0,
		1:  4.0 / 5.0,
		2:  -1.0 / 5.0,
		3:  4.0 / 105.0,
		4:  -1.0 / 280.0,
	}

	// get anlytic gradient of type model_mat and reshape it into 1 long vector
	gradientMat := model.dLoss(data, lambda)
	gradient := gradientMat.modelToTheta()

	// Test only for 100 elements and use a big prime to ensure a somewhat
	// random-like, but deterministic selection of indices
	theta := model.modelToTheta()
	for i := 0; i < 1; i++ {
		// index selection
		testIndex := ((i + 1) * 1299721) % theta.Len()

		// analytic gradient at test index
		analytic := gradient.AtVec(testIndex)

		// finite difference formula for test index
		// diff/fd package would also be suitable, but this way it's more simple
		fd := 0.0
		for d, w := range numDiff {
			fd += model.loss(data, lambda, map[int]float64{testIndex: d * h}) * w
		}
		fd /= h

		// compare and report if relative difference exceeds tol
		diff := math.Abs(analytic - fd)
		if diff < tol { // very small gradient
			continue
		}
		if diff/(math.Abs(analytic)+math.Abs(fd)) > tol {
			log.Printf("Theta element %d, with value %e, has finite difference gradient %e but analytic gradient %e. That looks like an error.\n",
				testIndex, theta.AtVec(testIndex), fd, analytic)
			//break
		}
	}
}

// getLoss acts on a model receiver and gets all input data and
// a flag to decide whether to print information while optimizing.
// It stores the loss with and without weight decay, as well as the
// classification error rate for training data, validation data and
// test data.
func (model *Model) getLoss(datas *inputdata.Datas, printAndPlot bool) {

	// slice of strings to iterate over (in order to keep the order
	// of the output the same for every run)
	s := []string{"training", "validation", "test"}

	// data map in order to iterate over the 3 data sets
	dataMap := map[string]*inputdata.Data{
		s[0]: datas.Training,
		s[1]: datas.Validation,
		s[2]: datas.Test,
	}

	// initialize a map with keys of type accuracy and collect all
	// losses and error rates
	model.FinalLossMap = make(map[string]accuracy, 3)

	for _, dataName := range s {
		data := dataMap[dataName]

		// loss with weight decay and error rate
		loss := model.loss(data, model.Lambda, nil)
		ac := accuracy{
			Loss:              loss,
			LossWithoutLambda: loss,
			ErrorRate:         model.errorRate(data),
		}
		if printAndPlot {
			fmt.Printf("\nThe loss on the %s data is %f\n", dataName, loss)
		}

		// overwrite lossWithoutLambda in case lambda is 0
		if model.Lambda != 0 {
			ac.LossWithoutLambda = model.loss(data, 0, nil)
			if printAndPlot {
				fmt.Printf("The classification loss (i.e. without weight decay) on the %s data is %f\n",
					dataName, ac.LossWithoutLambda)
			}
		}
		if printAndPlot {
			fmt.Printf("The classification error rate on the %s data is %f\n",
				dataName, ac.ErrorRate)
		}
		model.FinalLossMap[dataName] = ac
	}
}

// errorRate acts on a model receiver and gets a variable of type data as input.
// It returns the fraction of data cases that is incorrectly classified by the model.
func (model *Model) errorRate(data *inputdata.Data) float64 {
	_, numCases := data.Inputs.Dims()

	// get the index where to find the chosen class and compare
	// model with targets
	_, choices := matfunc.MaxPerRow(model.predict(data, numCases).classInput)
	_, targets := matfunc.MaxPerRow(data.Targets)
	_, sumNotEqual := slicefunc.EqualInt(choices, targets, false)
	return float64(sumNotEqual) / float64(numCases)
}

// PredictVersusTarget acts on a model receiver and gets data of type *inputdata.Data
// as input (supposed to be test data), and a maximum number of misclassified images
// to plot and report. It returns an error or nil.
func (model *Model) PredictVersusTarget(data *inputdata.Data) *MisclassImg {

	_, numCases := data.Inputs.Dims()
	// indices where the highest probability was found. Corresponds to a number 0-9
	_, choices := matfunc.MaxPerRow(model.predict(data, numCases).classInput)
	_, targets := matfunc.MaxPerRow(data.Targets)
	// id == true means misclassified
	idNotEqual, sum := slicefunc.EqualInt(choices, targets, false)

	misPred := make([]int, 0, sum)
	misTarget := make([]int, 0, sum)
	for i, id := range idNotEqual {
		if !id {
			continue
		}
		misPred = append(misPred, choices[i])
		misTarget = append(misTarget, targets[i])
	}
	return &MisclassImg{
		Predictions: misPred,
		Targets:     misTarget,
	}
}

// Plot plots the learning curve for the given model to the given io.Writer
func (model *Model) Plot(w io.Writer) error {
	return plotLearningCurve(w, model.lossPerIter)
}

// checkIfBest acts on a model receiver and gets the so far best model
// of type safemodel as input. In case the new model is better, it is copied
// into the best one. The mutex of the safemodel is used to prevent parallel checking.
func (model *Model) checkIfBest(bestModel *safemodel) {
	if model == nil {
		return
	}

	bestModel.Lock()
	defer bestModel.Unlock()
	bestLoss := bestModel.FinalLossMap["validation"].LossWithoutLambda
	if l := model.FinalLossMap["validation"].LossWithoutLambda; l < bestLoss {
		bestModel.Model = model.copy()
	}
}
