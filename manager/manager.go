package manager

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/bschieche/gomnist/mnist"
	"github.com/bschieche/gomnist/utils/inputdata"
)

// Manager stores all relevant information to learn models
// with given parameter lists. It especially stores the resulting
// models in a map specified by unique id's (coming from counter)
type Manager struct {
	sync.Mutex
	jobs    chan Job
	Datas   *inputdata.Datas
	Results map[string]*mnist.Model
	counter int
}

// Job consists of a parameter list (= input) and a unique id
type Job struct {
	Input mnist.CrossvalList
	ID    string
}

// New loads the data, initializes the manager, starts
// the run method in a goroutine and returns the manager
func New() *Manager {

	datas := &inputdata.Datas{
		FilenameTrainingInput:    "data",
		FilenameTrainingTarget:   "data",
		FilenameTestInput:        "data",
		FilenameTestTarget:       "data",
		FilenameValidationInput:  "data",
		FilenameValidationTarget: "data",
	}
	datas.LoadMNISTdata()

	m := &Manager{
		Datas:   datas,
		jobs:    make(chan Job, 10),
		Results: make(map[string]*mnist.Model),
		counter: 1,
	}
	go m.run()
	return m
}

// Submit is called from the controller and inputs a list
// of hyperparameter combination (of type mnist.CrossvalList).
// It returns the unique id for that specific input
func (m *Manager) Submit(hps mnist.CrossvalList) string {
	// extract current id from manager and increase counter by 1
	id := strconv.Itoa(m.counter)
	m.counter++

	// fill channel with a new Job with the current input and the id
	m.jobs <- Job{
		ID:    id,
		Input: hps,
	}
	return id
}

// The run method is called from the New method in a goroutine and
// performs the actual learning of the crossvalidation parameter list
// given by the Job extracted from the Job channel. The resulting
// best model is stored in the results map.
func (m *Manager) run() {
	for job := range m.jobs {
		model, err := job.Input.ConcurrentCrossVal(m.Datas)
		if err != nil {
			fmt.Println(err)
		}
		m.Results[job.ID] = model
	}
}
