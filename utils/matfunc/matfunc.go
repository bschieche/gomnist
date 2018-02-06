// Package matfunc provides helper functions for matrices and vectors
// of the gonum/mat package. Functions do not change input matrices.
package matfunc

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// PrintMatrix gets a matrix and two integers indicating the number of
// rows and columns to print.
func PrintMatrix(mt mat.Matrix, maxi, maxj int) {
	// prints the matrix with maxi rows and maxj columns
	logger := log.New(os.Stdout, "!!Warning PrintMatrix: ", 0)
	n, m := mt.Dims()
	if n < maxi {
		logger.Println(mat.ErrRowAccess)
		return
	}
	if m < maxj {
		logger.Println(mat.ErrColAccess)
		return
	}
	for i := 0; i < maxi; i++ {
		for j := 0; j < maxj; j++ {
			fmt.Printf("%+.4e ", mt.At(i, j))
		}
		fmt.Print("\n")
	}
	fmt.Print("\n")
}

// SliceCont is a version of the mat.Slice method, where k and
// l may exceed the dimensions of the matrix a. The indices are
// restarted with 0 again. i and j still need to be in the range
// of the rows and columns and k and l have the restriction that
// they have to be smaller than 2 x the row and column number,
// respectively. If this is the case, SliceCont panics.
func SliceCont(a *mat.Dense, i, k, j, l int) *mat.Dense {

	n, m := a.Dims()
	if i >= n || j >= m || k > 2*n || l > 2*m {
		panic(mat.ErrIndexOutOfRange)
	}

	b := mat.NewDense(k-i, l-j, nil)

	switch {
	case k <= n && l <= m: // normal slicing
		b = a.Slice(i, k, j, l).(*mat.Dense)
	case k > n && l <= m: // start with top row again
		b.Stack(a.Slice(i, n, j, l), a.Slice(0, k-n, j, l))
	case k <= n && l > m: // start with first column again
		b.Copy(Compose(a.Slice(i, n, j, m), a.Slice(i, n, 0, l-m)))
	case k > n && l > m: // start with top row and first column again
		c1 := mat.NewDense(k-i, m-j, nil)
		c1.Stack(a.Slice(i, n, j, m), a.Slice(0, k-n, j, m))
		c2 := mat.NewDense(k-i, l-m, nil)
		c2.Stack(a.Slice(i, n, 0, l-m), a.Slice(0, k-n, 0, l-m))
		b.Copy(Compose(c1, c2))
	}
	return b
}

// Compose is the same as mat.Stack, but horizontally
func Compose(a, b mat.Matrix) mat.Matrix {
	n1, m1 := a.Dims()
	n2, m2 := b.Dims()
	if n1 != n2 {
		panic(mat.ErrShape)
	}
	c := mat.NewDense(m1+m2, n1, nil)
	c.Stack(a.T(), b.T())
	return c.T()
}

// LogisticMat gets a matrix as input and returns the elemenentwise
// logistic function as a *mat.Dense.
func LogisticMat(a mat.Matrix) *mat.Dense {
	n, m := a.Dims()
	b := mat.NewDense(n, m, nil)
	fn := func(i, j int, v float64) float64 {
		return logistic(v)
	}
	b.Apply(fn, a)
	return b
}

// unexported logistic function
func logistic(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// LogisticGradMat gets a matrix as input and returns the elemenentwise
// derivative of the logistic function as a *mat.Dense.
func LogisticGradMat(a mat.Matrix) *mat.Dense {
	n, m := a.Dims()
	b := mat.NewDense(n, m, nil)
	fn := func(i, j int, v float64) float64 {
		return logisticGrad(v)
	}
	b.Apply(fn, a)
	return b
}

// unexported derivative of the logistic function
func logisticGrad(v float64) float64 {
	g := logistic(v)
	return g * (1.0 - g)
}

// unexported derivative of logistic function to compute log(sum(exp(a)))
// columnwise in a numerically stable way
func logSumExpOverRows(a mat.Matrix) mat.Vector {
	n, m := a.Dims()
	processedA := mat.NewVecDense(m, nil)
	aCol := make([]float64, n)
	// extract each column as a slice, sort it to get the largest value,
	// and then compute the exponential of each column normalized by
	// the maximum column value; finally take the logarithm again and
	// denormalize with the maximum column value
	for j := 0; j < m; j++ {
		aCol = mat.Col(aCol, j, a)
		sort.Float64s(aCol)
		maxACol := aCol[len(aCol)-1]
		b := 0.0
		for _, aij := range aCol {
			b += math.Exp(aij - maxACol)
		}
		processedA.SetVec(j, math.Log(b)+maxACol)
	}
	return processedA
}

// LogProb gets a matrix as input and computes the log probabilities of
// a softmax.
func LogProb(a mat.Matrix) *mat.Dense {
	normalizer := logSumExpOverRows(a)
	return AddScaledVectorToMatrix(a, normalizer, -1, 2)
}

// AddScaledVectorToMatrix gets a matrix a, a vector v, a scaling factor scale and a dimension d
// (default is rowwise, d=2 indicates columnwise) and returns a+scale*v as a *mat.Dense.
func AddScaledVectorToMatrix(a mat.Matrix, v mat.Vector, scale float64, d int) *mat.Dense {
	n, m := a.Dims()
	l := v.Len()
	if m != l {
		panic(mat.ErrShape)
	}
	fn := func(i, j int, c float64) float64 {
		if d == 2 {
			return a.At(i, j) + scale*v.AtVec(j)
		}
		return a.At(i, j) + scale*v.AtVec(i)
	}
	b := mat.NewDense(n, m, nil)
	b.Apply(fn, a)
	return b

}

// ExpElem gets a matrix as input and returns a *mat.Dense of the elementwise
// exponential.
func ExpElem(a mat.Matrix) *mat.Dense {
	n, m := a.Dims()
	b := mat.NewDense(n, m, nil)
	b.Apply(func(i, j int, v float64) float64 { return math.Exp(a.At(i, j)) }, a)
	return b
}

// SumElem gets two *mat.Dense matrices, multiplies them elementwise and
// returns the sum.
func SumElem(a, b *mat.Dense) float64 {
	n1, m1 := a.Dims()
	n2, m2 := b.Dims()
	if n1 != n2 || m1 != m2 {
		panic(mat.ErrShape)
	}
	m := mat.NewDense(n1, m1, nil)
	m.MulElem(a, b)
	return mat.Sum(m)
}

// SquareSum gets a vector as input and returns the squared sum.
func SquareSum(v mat.Vector) float64 {
	v2 := mat.NewVecDense(v.Len(), nil)
	v2.MulElemVec(v, v)
	return mat.Sum(v2)
}

// MaxPerRow gets a input matrix of type *mat.Dense and returns
// the maximum for each columns as a vector together with the
// indices as an int slice of same length.
func MaxPerRow(a *mat.Dense) (mat.Vector, []int) {
	_, m := a.Dims()
	maxPerRow := mat.NewVecDense(m, nil)
	idMax := make([]int, m)
	for j := 0; j < m; j++ {
		max, id := Argmax(a.ColView(j))
		maxPerRow.SetVec(j, max)
		idMax[j] = id
	}
	return maxPerRow, idMax
}

// Argmax gets a vector as input and returns the index of the maximum
// entry together with the maximum.
func Argmax(v mat.Vector) (float64, int) {
	max := v.AtVec(0)
	id := 0
	for i := 1; i < v.Len(); i++ {
		c := v.AtVec(i)
		if c <= max {
			continue
		}
		max = c
		id = i
	}
	return max, id
}
