"""Photometric conversions used before segmentation.

Two representations of the same RGB image are required later:

- Luma (grayscale) is the scalar field whose Fourier transform is taken.
  A spectrum of a 3-channel array is not well defined in the textbook
  2D DFT, so the image is reduced to one channel first.
- HSV separates chromatic angle from intensity. Water color is unstable
  in RGB (a lake can be blue, green, brown, or nearly grey) but it often
  stays inside a limited hue interval while brightness changes with depth
  and sky reflection.

Both conversions are written out with NumPy. No color-conversion routine
from an image-processing library is called.
"""

from __future__ import annotations

import numpy as np

# ITU-R BT.601 luma coefficients. They pre-date the HDTV BT.709 weights
# (0.2126, 0.7152, 0.0722) and remain the coefficients used in most digital
# image processing textbooks when a display-referred luminance is needed.
_LUMA_R = np.float32(0.299)
_LUMA_G = np.float32(0.587)
_LUMA_B = np.float32(0.114)


def as_float01(rgb: np.ndarray) -> np.ndarray:
    """Return RGB as float32 in [0, 1] without copying if it already is.

    uint8 images are divided by 255. Values already in [0, 1] are cast to
    float32 only. This convention is used everywhere else so that a
    formula does not silently change meaning between 0–255 and 0–1.
    """
    array = np.asarray(rgb)
    if array.dtype == np.uint8:
        return array.astype(np.float32) * np.float32(1.0 / 255.0)
    out = array.astype(np.float32, copy=False)
    return out


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to luma with the BT.601 weighted sum.

    Y(x, y) = 0.299 R + 0.587 G + 0.114 B

    The weights are positive and sum to 1, so Y stays in [0, 1] when the
    channels do. Green dominates because the eye is most sensitive there;
    treating the three channels equally would under-represent that.

    Returns a float32 array of shape (H, W) with values in [0, 1].
    """
    image = as_float01(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    # Einsum keeps the weighted sum explicit and avoids stacking temporaries.
    return (
        _LUMA_R * image[..., 0] + _LUMA_G * image[..., 1] + _LUMA_B * image[..., 2]
    ).astype(np.float32, copy=False)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV with the standard hexcone definition.

    Let R, G, B be in [0, 1] and write

        Cmax = max(R, G, B)
        Cmin = min(R, G, B)
        Δ    = Cmax − Cmin

    Value and saturation:

        V = Cmax
        S = 0            if Cmax = 0
        S = Δ / Cmax     otherwise

    Hue, in degrees, is the angle around the hexcone. It is undefined when
    Δ = 0 (a grey pixel); the implementation stores 0, which is only a
    convention and is harmless because S = 0 there too.

        H = 60° · (((G − B) / Δ) mod 6)     if Cmax = R
        H = 60° · (((B − R) / Δ) + 2)       if Cmax = G
        H = 60° · (((R − G) / Δ) + 4)       if Cmax = B

    The modulo on the red sector folds negative angles (magenta, near 0°)
    back into [0, 360).

    Why HSV is used for water
    -------------------------
    A threshold on blue in RGB rejects green, turbid, and dark water.
    Hue is invariant to a uniform scaling of (R, G, B), so a deeper copy
    of the same water color keeps approximately the same H. Saturation
    then separates chromatic water from grey pavement, and value separates
    it from white cloud and foam when the user sets those gates.

    Returns float32 of shape (H, W, 3): H in [0, 360), S in [0, 1], V in [0, 1].
    """
    image = as_float01(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]

    cmax = np.maximum(np.maximum(red, green), blue)
    cmin = np.minimum(np.minimum(red, green), blue)
    delta = cmax - cmin

    hue = np.zeros(red.shape, dtype=np.float32)
    # Pixels with Δ = 0 stay at hue 0. The divisions below are masked.
    nonzero = delta > 1e-8

    red_max = nonzero & (cmax == red)
    green_max = nonzero & (cmax == green)
    blue_max = nonzero & (cmax == blue)

    # ((G - B) / Δ) mod 6, then scaled to degrees. np.mod handles negatives.
    hue[red_max] = (60.0 * np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)).astype(
        np.float32
    )
    hue[green_max] = (60.0 * (((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0)).astype(
        np.float32
    )
    hue[blue_max] = (60.0 * (((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0)).astype(
        np.float32
    )
    # Numerical noise can leave a value a fraction above 360. Fold it.
    np.mod(hue, 360.0, out=hue)

    saturation = np.zeros(red.shape, dtype=np.float32)
    chromatic = cmax > 1e-8
    saturation[chromatic] = (delta[chromatic] / cmax[chromatic]).astype(np.float32)

    hsv = np.empty(image.shape, dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = np.clip(saturation, 0.0, 1.0)
    hsv[..., 2] = np.clip(cmax, 0.0, 1.0)
    return hsv


def gray_to_uint8(gray: np.ndarray) -> np.ndarray:
    """Map a [0, 1] luminance image to uint8 for display."""
    return np.clip(np.round(np.asarray(gray, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate luma into three channels so every preview widget is RGB."""
    g = gray_to_uint8(gray) if gray.dtype != np.uint8 else gray
    if g.ndim == 3:
        return g
    return np.stack((g, g, g), axis=-1)
