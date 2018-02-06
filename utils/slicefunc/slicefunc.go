// Package slicefunc provides helper functions for slices
package slicefunc

// IntSliceProd gets an int slice as input and returns the product of all elements
func IntSliceProd(vals []int) int {
	if vals == nil {
		return 0
	}
	prod := 1
	for _, v := range vals {
		prod *= v
	}
	return prod
}

// GetCombis gets the slice of int slices to fill in the results,
// an int slice indicating the number of combinations for each dimension,
// the dimension to fill and the total number of combinations
// It's important that the results slice is initialized with the correct
// length n and the inner int slices need to be nil
func GetCombis(combis [][]int, lc []int, dim, n int) {

	// return when dim exceeds dimensions
	if dim == len(lc) {
		return
	}

	counter := 0
	// repeat all combinations until n is reached
	for {
		for i := 0; i < lc[dim]; i++ {
			combis[counter] = append(combis[counter], i)
			counter++
		}
		if counter == n {
			break
		}
	}

	GetCombis(combis, lc, dim+1, n)
}

// EqualInt gets two int slices of the same length (if not it panics) and a bool
// returns slice of bools of the same length with true for equality and false else if equal is true, and the corresponding sum (= number of equals)
// returns slice of bools with false for equality and true else if equal is false, and the corresponding sum (= number of non-equals)
func EqualInt(v1, v2 []int, equal bool) ([]bool, int) {
	l := len(v1)
	if len(v2) != l {
		panic("The two slices need to have the same length.")
	}

	v := make([]bool, l)
	sum := 0
	for i, value := range v1 {
		if equal && value == v2[i] || !equal && value != v2[i] {
			v[i] = true
			sum++
		}
	}

	return v, sum
}
