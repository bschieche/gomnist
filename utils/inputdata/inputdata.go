// Package inputdata provides some functions to load csv files into a slice or a matrix.
// It's especially designed for supervised machine learning data, which are split
// up into training data, validation data, and test data, each consisting of input data
// and target data. The package also contains functions to load the official MNIST data
// set.
package inputdata

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
)

// Data stores input and target data
type Data struct {
	Inputs  *mat.Dense
	Targets *mat.Dense
}

// Datas stores training data, validation data, and test data, each pointing
// to a variable of type Data together with the filenames
type Datas struct {
	Training                 *Data
	Validation               *Data
	Test                     *Data
	FilenameTrainingInput    string
	FilenameValidationInput  string
	FilenameTestInput        string
	FilenameTrainingTarget   string
	FilenameValidationTarget string
	FilenameTestTarget       string
}

// LoadDatas collects training data, validation data, and test data into its receiver
// of type *Datas; it expects filenames for all input and target data stored already
// in the receiver; otherwise the functions does nothing.
func (datas *Datas) LoadDatas() {
	datas.Training = &Data{}
	datas.Test = &Data{}
	datas.Validation = &Data{}
	datas.Training.LoadData(datas.FilenameTrainingInput, datas.FilenameTrainingTarget)
	datas.Test.LoadData(datas.FilenameTestInput, datas.FilenameTestTarget)
	datas.Validation.LoadData(datas.FilenameValidationInput, datas.FilenameValidationTarget)
}

// LoadData collects input and target in matrix format into the receiver of type *Data in case filenames
// are given; otherwise it does nothing. This enables the function to be used for unsupervised
// learning where only input data is available.
func (data *Data) LoadData(inputfile, targetfile string) {
	if inputfile != "" {
		inputs, err := GetDataAsMatrix(inputfile)
		if err != nil {
			log.Fatal(err)
		}

		data.Inputs = inputs
	}
	if targetfile != "" {
		targets, err := GetDataAsMatrix(targetfile)
		if err != nil {
			log.Fatal(err)
		}
		data.Targets = targets
	}
}

// GetDataAsMatrix gets a filename and returns a matrix and nil or nil and an error.
// Similar to GetDataAsSlice
func GetDataAsMatrix(filename string) (*mat.Dense, error) {
	records, err := LoadCSV(filename)
	if err != nil {
		return nil, fmt.Errorf("Error with %q: %s\n", filename, err)
	}

	nrow := len(records)
	if nrow == 0 {
		return nil, fmt.Errorf("Empty rows for %q\n", filename)
	}

	ncol := len(records[0])
	if ncol == 0 {
		return nil, fmt.Errorf("Empty columns for %q\n", filename)
	}

	// parse records read from csv into a matrix
	data := mat.NewDense(nrow, ncol, nil)
	for ir, record := range records {
		if len(record) != ncol {
			return nil, fmt.Errorf("Missing data or inconsistent lengths of rows for %q\n", filename)
		}
		for ic, number := range record {
			if s, err := strconv.ParseFloat(number, 64); err == nil {
				data.Set(ir, ic, s)
			}
		}
	}

	return data, nil
}

// GetDataAsSlice gets a filename and returns a long data slice and nil or nil and an error.
// Similar to GetDataAsMatrix
func GetDataAsSlice(filename string) ([]float64, error) {
	records, err := LoadCSV(filename)
	if err != nil {
		return nil, fmt.Errorf("Error with %q: %s\n", filename, err)
	}

	nrow := len(records)
	if nrow == 0 {
		return nil, fmt.Errorf("Empty rows for %q\n", filename)
	}

	ncol := len(records[0])
	if ncol == 0 {
		return nil, fmt.Errorf("Empty columns for %q\n", filename)
	}

	longrecords := make([]float64, 0, nrow*ncol)
	for _, record := range records {
		for _, number := range record {
			if len(record) != ncol {
				return nil, fmt.Errorf("Missing data or inconsistent lengths of rows for %q\n", filename)
			}
			if s, err := strconv.ParseFloat(number, 64); err == nil {
				longrecords = append(longrecords, s)
			}
		}
	}

	return longrecords, nil
}

// LoadCSV loads a csv file given in filename and returns a slice of string slices as records and nil
// or nil and an error
func LoadCSV(filename string) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("Error: %s\n", err)
	}

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("Error: %s\n", err)
	}

	return records, nil
}

// LoadMNISTdata loads the MNIST data set from http://yann.lecun.com/exdb/mnist/
// and stores it in a datas struct.
func (datas *Datas) LoadMNISTdata() error {

	// initialize data sets in the datas struct
	datas.Training = &Data{}
	datas.Test = &Data{}
	datas.Validation = &Data{}

	// load data as *GoMNIST.Set variables
	train, test, err := GoMNIST.Load(datas.FilenameTrainingTarget)
	if err != nil {
		return err
	}

	// split training set into training set and validation set
	train, validation, err := Split(train, 0.8)
	if err != nil {
		return err
	}

	// transform data into matrices and store it in the datas struct
	datas.Training.FillData(train)
	datas.Validation.FillData(validation)
	datas.Test.FillData(test)

	return nil
}

// Split gets a data set and splits it into two data sets.
// Fraction indicates the proportion of the orginial data
// to keep. The rest is put in the second one.
// The function returns the two data sets and nil or nil and an error.
// The function is especially designed for splitting training data into
// training and validation data. It only makes sense if the input data
// is sufficiently shuffeled.
func Split(dataset *GoMNIST.Set, fraction float64) (*GoMNIST.Set, *GoMNIST.Set, error) {

	// error handling
	if fraction < 0 || fraction > 1 {
		return nil, nil, fmt.Errorf("fraction has to be a real number between 0 and 1")
	}

	// number of images
	numImages := dataset.Count()
	// index for splitting
	nSplit := int(math.Ceil(float64(numImages) * fraction))

	// generate the two new sets
	set1 := &GoMNIST.Set{
		NRow:   dataset.NRow,
		NCol:   dataset.NCol,
		Images: dataset.Images[:nSplit],
		Labels: dataset.Labels[:nSplit],
	}

	set2 := &GoMNIST.Set{
		NRow:   dataset.NRow,
		NCol:   dataset.NCol,
		Images: dataset.Images[nSplit:],
		Labels: dataset.Labels[nSplit:],
	}

	return set1, set2, nil
}

// FillData acts on a data struct and gets input data
// of the type *GoMNIST.Set. It restores these as
// matrices into the data struct.
func (data *Data) FillData(dataset *GoMNIST.Set) {

	// number of images
	numImages := dataset.Count()

	// initialize input and target matrices
	data.Inputs = mat.NewDense(dataset.NRow*dataset.NCol, numImages, nil)
	data.Targets = mat.NewDense(10, numImages, nil)

	// fill input matrix with grayscale color values between 0 and 1
	for i, image := range dataset.Images {
		for j, color := range image {
			data.Inputs.Set(j, i, float64(color)/255)
		}
	}

	// fill target matrix with one hot encoded labels
	// insted of absolute labels
	for i, label := range dataset.Labels {
		data.Targets.Set(int(label), i, 1)
	}
}
