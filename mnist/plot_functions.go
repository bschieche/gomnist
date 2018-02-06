package mnist

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"

	"github.com/nfnt/resize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// plotLearningCurve gets a slice of iteration numbers, a slice of training data losses
// of the same length, a slice of validation data losses of the same length, and a
// path where to save the plot. The learning curve is plotted and saved as a png. If
// an error occurs it is passed on to the caller
func plotLearningCurve(w io.Writer, losses lossPerIter) error {

	// new plot
	p, err := plot.New()
	if err != nil {
		return err
	}

	// some settings for axis labels, title, and legend
	p.Title.Text = "Learning curve"
	p.X.Label.Text = "iteration number"
	p.Y.Label.Text = "loss"
	p.Legend.Top = true

	// transform data into plotter.XYs
	ptsTraining, err := makePlotter(losses.iters, losses.trainingLosses)
	if err != nil {
		return err
	}
	ptsValidation, err := makePlotter(losses.iters, losses.validationLosses)
	if err != nil {
		return err
	}

	// plot the curves
	err = plotutil.AddLines(p,
		"Training", ptsTraining,
		"Validation", ptsValidation,
	)
	if err != nil {
		return err
	}

	// Save the plot to a PNG file.
	wt, err := p.WriterTo(4*vg.Inch, 4*vg.Inch, "png")
	if err != nil {
		return err
	}
	_, err = wt.WriteTo(w)
	return err
}

// makePlotter gets two slices and returns the corresponding plotter.XYs
// and nil or nil and an error
func makePlotter(x, y []float64) (plotter.XYs, error) {

	// error handling
	if len(x) != len(y) {
		return nil, fmt.Errorf("x and y point slices need to have the same length for scatter plot")
	}

	// store values as slice of points = plotter.XYs
	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}

	return pts, nil
}

// PrintImage gets a slice of grayscale values between 0 and 255 of length 16x16=256
// and a path where to save the image. The grayscale image is filled with the values
// and resized to 100 pixels. Then it's saved. It returns nil or and error when
// something goes wrong.
func PrintImage(w io.Writer, imageData []float64) error {

	px := int(math.Sqrt(float64(len(imageData))))

	img := image.NewGray(image.Rect(0, 0, px, px))
	var imgScaled image.Image

	// fill image column for column
	for k, c := range imageData {
		if c < 0 || c > 1 {
			return fmt.Errorf("Gray scale value below or above 1. Check data!\n")
		}
		// color.Gray ranges from 0 to 255, 0 is black, 255 is white
		img.SetGray(k%px, k/px, color.Gray{uint8(255 * c)})
	}

	// scale image to 100 pixels
	imgScaled = resize.Resize(100, 0, img, resize.Lanczos3)

	return png.Encode(w, imgScaled)
}
