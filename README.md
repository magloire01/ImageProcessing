# WaterVision

Desktop application for detecting natural water in a photograph, cleaning the mask, and estimating the water surface. It is an academic digital-image-processing project. The arithmetic is NumPy and SciPy. The window is CustomTkinter. Images are loaded with Pillow and charts are drawn with Matplotlib.

OpenCV is not used.

## Run

```text
python -m pip install -r requirements.txt
python main.py
```

Python 3.11 or newer is required. The window starts on the dashboard. Load a JPG, PNG, TIFF, or BMP file, or open the built-in demonstration scene. That scene contains a smooth blue-green lake, a river, a dark grey pond, and a smooth tan sandbar. The sandbar is there so the frequency method can be seen accepting a dry surface.

## What each module does

Image acquisition reads the file, applies the EXIF orientation, and reports width, height, and channel depth. Zoom is a single scale factor, so the picture is never stretched.

Preprocessing is optional and is applied to the original pixels, not stacked on a previous preview. The order is median, then Gaussian, then contrast. The median removes impulses before a linear filter can spread them. The Gaussian is the sampled kernel

```text
G(x, y) = 1 / (2 π σ²) · exp(−(x² + y²) / (2 σ²))
```

Contrast is a percentile stretch, a gamma power, or a luma histogram equalization that rescales RGB so the hue is kept. Analysis uses the image only after Apply.

Fourier analysis computes the 2D discrete Fourier transform of BT.601 luma, shifts the zero frequency to the center, and displays `log(1 + |F|)`. The logarithm is for display. Filters multiply the complex coefficients. A Butterworth low-pass of order n is

```text
H(u, v) = 1 / (1 + (D(u, v) / D0) ^ (2 n))
```

D is the distance to the spectrum center and D0 is a fraction of the corner radius. The high-pass is the classical complement, with the center sample set to 0 because `D0 / D` is undefined there. An ideal disk is available as the textbook definition; it rings more than the Butterworth.

Local frequency analysis repeats that idea on Hann-windowed patches. After the patch mean is removed, the high-frequency ratio is the share of spectral energy outside a disk. The map is resampled to the image only so later rules can read a value per pixel. A low value means a smooth neighborhood. It does not mean water. Sand, pavement, and shadow are smooth too.

Segmentation has three masks.

1. HSV. A pixel is inside a hue interval (the interval may cross 0°), a saturation range, and a value range. The default interval is blue only, from 185° to 255°. A second ochre band and a dark-water rule can be turned on, and both are off by default.
2. Fourier. The pixel is a candidate where the local high-frequency ratio is below a threshold. The mask is labeled as a candidate.
3. Hybrid. The score is a weighted sum of soft HSV membership, a logistic smoothness term centered on the same threshold, and local homogeneity from the box-filter variance `E[X²] − (E[X])²`. Weights are normalized so they sum to 1. With the defaults, smoothness and homogeneity together stay below the decision threshold, so frequency evidence cannot carry the label by itself.

Morphology uses a disk built in NumPy. Opening is erosion followed by dilation. Closing is the reverse. Opening runs first so thin false positives are deleted before closing can join them. Enclosed holes are filled. Components smaller than a pixel count are removed. Connectivity is 8.

Area is

```text
water percentage = (water pixels / image pixels) × 100
```

Every pixel of the working image is counted. A value in square meters is

```text
water pixels × pixel width (m) × pixel height (m)
```

and it is reported only when calibration is turned on. Otherwise the interface says the result is a pixel estimate, not a ground area. The scale is assumed constant and is not corrected for slope.

Evaluation compares each cleaned mask with a binary reference. IoU, precision, recall, F1, and pixel accuracy are reported. F1 equals the Dice coefficient for one binary class. Pixel accuracy includes true negatives, so it stays high when most of the frame is dry; IoU and F1 describe the water region. A reference of a different size is resized with nearest neighbor, which does not invent grey levels in a label image, and the report says that this happened.

Export writes a 0/255 mask, a highlighted image, the spectrum picture, a CSV table, and a PDF report.

## Checks

```text
python -m unittest discover -s tests -t .
```

The demonstration-scene test checks that the hybrid mask covers the lake, that the Fourier mask accepts the sandbar, and that the hybrid mask does not treat that sandbar as water.
