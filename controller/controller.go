package controller

import (
	"fmt"
	"html/template"
	"math/rand"
	"net/http"
	"strconv"
	"strings"

	"github.com/bschieche/gomnist/manager"
	"github.com/bschieche/gomnist/mnist"
	"github.com/go-kit/kit/log"
	"github.com/go-kit/kit/log/level"
	"gonum.org/v1/gonum/mat"
)

// Controller has a logger capable for logging and a manager
// that starts the calculation and stores the results
type Controller struct {
	Log     log.Logger
	Manager *manager.Manager
}

// New gets a logger and a manager and returns
// a controller.
func New(l log.Logger, m *manager.Manager) *Controller {
	if l == nil {
		l = log.NewNopLogger()
	}
	return &Controller{
		Log:     l,
		Manager: m,
	}
}

// template data: parameters for the html template that the program
// generates.
type tplData struct {
	ID           string
	Lambda       string
	NumHidden    string
	NumIters     string
	LearningRate string
	Momentum     string
	BatchSize    string
	Loss         string
	ErrorRate    string
}

// ServeIndex runs when the browser sends a request from /,
// specified in the routes in the main.
func (c *Controller) ServeIndex(w http.ResponseWriter, r *http.Request) {
	// parse the template
	t, err := template.New("index").Parse(pageTemplate)
	if err != nil {
		level.Error(c.Log).Log("msg", "template error", "err", err)
		http.Error(w, "Template error", http.StatusInternalServerError)
		return
	}

	id := r.URL.Query().Get("id")
	lambda := ""
	numHidden := ""
	numIters := ""
	learningRate := ""
	momentum := ""
	batchSize := ""
	loss := ""
	errorRate := ""
	model, found := c.Manager.Results[id]
	if found || model != nil {
		lambda = fmt.Sprintf("%.4e", model.Lambda)
		numHidden = fmt.Sprintf("%d", model.NumHidden)
		numIters = fmt.Sprintf("%d", model.NumIters)
		learningRate = fmt.Sprintf("%.4e", model.LearningRate)
		momentum = fmt.Sprintf("%.4e", model.Momentum)
		batchSize = fmt.Sprintf("%d", model.BatchSize)
		loss = fmt.Sprintf("%.4e", model.FinalLossMap["validation"].LossWithoutLambda)
		errorRate = fmt.Sprintf("%.4e", model.FinalLossMap["validation"].ErrorRate)

	}

	// fill table data with id and best hyperparameters in case a solution is
	// already available.
	td := tplData{
		ID:           id,
		Lambda:       lambda,
		NumHidden:    numHidden,
		NumIters:     numIters,
		LearningRate: learningRate,
		Momentum:     momentum,
		BatchSize:    batchSize,
		Loss:         loss,
		ErrorRate:    errorRate,
	}

	// Apply parsed template to the template data, and write the output to the writer.
	if err := t.Execute(w, td); err != nil {
		level.Error(c.Log).Log("msg", "template error", "err", err)
	}
}

// ServeSubmit runs when the browser sends a request from /submit,
// specified in the routes in the main.
func (c *Controller) ServeSubmit(w http.ResponseWriter, r *http.Request) {

	// parse input parameters
	hps, err := processRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		fmt.Println("ERROR:", err)
		return
	}

	// put read parameter list into the Job channel. The Manager.Run
	// method awaits these jobs and executes them sequentially.
	// The returned unique id is used to redirect the browser.
	id := c.Manager.Submit(hps)
	u := r.URL
	u.Path = "/"
	values := u.Query()
	values.Set("id", id)
	u.RawQuery = values.Encode()
	http.Redirect(w, r, u.String(), http.StatusFound)
}

// ServeRender runs when the browser sends a request form /render,
// specified in the routes in the main.
func (c *Controller) ServeRender(w http.ResponseWriter, r *http.Request) {
	// get the parameters for the template
	id := r.URL.Query().Get("id")
	typ := r.URL.Query()["typ"][0]

	// if a solution model is found for the given id the model starts
	// a plot method to plot both the learning rate and a misclassified
	// image.
	model, found := c.Manager.Results[id]
	if !found || model == nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "image/png")
	switch typ {
	case "learningCurve":
		model.Plot(w)
	case "digit":
		misclassImg := model.PredictVersusTarget(c.Manager.Datas.Test)
		i := rand.Intn(len(misclassImg.Predictions))
		fmt.Printf("Test case %d is \"%d\", but was incorrectly classified as \"%d\".\n",
			i+1, misclassImg.Targets[i], misclassImg.Predictions[i])

		mnist.PrintImage(w, mat.Col(nil, i, c.Manager.Datas.Test.Inputs))
	}
}

// helper function to parse the input parameters of the form
func processRequest(r *http.Request) (mnist.CrossvalList, error) {
	hps := mnist.CrossvalList{EarlyStopping: []bool{true}}

	lambdas, err := parseFloats(r, "lambda")
	if err != nil {
		return hps, err
	}
	hps.Lambda = lambdas

	numHidden, err := parseInts(r, "numHidden")
	if err != nil {
		return hps, err
	}
	hps.NumHidden = numHidden

	numIters, err := parseInts(r, "numIters")
	if err != nil {
		return hps, err
	}
	hps.NumIters = numIters

	learningRate, err := parseFloats(r, "learningRate")
	if err != nil {
		return hps, err
	}
	hps.LearningRate = learningRate

	momentum, err := parseFloats(r, "momentum")
	if err != nil {
		return hps, err
	}
	hps.Momentum = momentum

	batchSize, err := parseInts(r, "batchSize")
	if err != nil {
		return hps, err
	}
	hps.BatchSize = batchSize

	return hps, nil
}

// helper function to parse floats as input parameters of the form
func parseFloats(r *http.Request, parName string) ([]float64, error) {
	text := r.FormValue(parName)
	// use default
	if text == "" {
		return []float64{0}, nil
	}

	var parameters []float64
	text = strings.Replace(text, ",", " ", -1)
	for _, field := range strings.Fields(text) {
		p, err := strconv.ParseFloat(field, 64)
		if err != nil {
			return parameters, fmt.Errorf("'" + field + "' is invalid")
		}
		parameters = append(parameters, p)
	}
	return parameters, nil
}

// helper function to parse ints as input parameters of the form
func parseInts(r *http.Request, parName string) ([]int, error) {
	text := r.FormValue(parName)
	// use default
	if text == "" {
		return []int{0}, nil
	}

	var parameters []int
	text = strings.Replace(text, ",", " ", -1)
	for _, field := range strings.Fields(text) {
		p, err := strconv.ParseInt(field, 10, 64)
		if err != nil {
			return parameters, fmt.Errorf("'" + field + "' is invalid")
		}
		parameters = append(parameters, int(p))
	}
	return parameters, nil
}
