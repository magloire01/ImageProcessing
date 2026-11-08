"""Spatial preprocessing: noise reduction and contrast.

Order of operations, and why it is this order
----------------------------------------------
1. Median filter. Salt-and-pepper impulses are replaced by a local order
   statistic before any linear filter can spread them into a halo.
2. Gaussian filter. What remains is treated as additive high-frequency
   noise and attenuated by convolution with a sampled Gaussian.
3. Contrast. Stretching, gamma, or histogram equalization is applied last
   so it does not amplify the noise the two filters just removed.

Each stage is optional. With the identity parameters the original array is
returned unchanged (a copy, so the caller can store it safely).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from watervision.processing.color import as_float01, rgb_to_gray


def _odd_kernel(size: int) -> int:
    """Median kernels are odd so the window has a unique center pixel."""
    size = int(size)
    if size < 1:
        return 1
    if size % 2 == 0:
        size += 1
    return min(size, 15)


def apply_median(rgb: np.ndarray, size: int) -> np.ndarray:
    """Replace each channel by the median of a square neighborhood.

    For a window W(x, y) the median m satisfies

        |{p in W : p ≤ m}| ≥ |W| / 2
        and
        |{p in W : p ≥ m}| ≥ |W| / 2.

    Unlike the mean, one extreme sample cannot pull m across the window.
    That is why the median removes impulse noise while preserving steps
    (shorelines) better than a box or Gaussian of the same width.

    The kernel size along the channel axis is 1, so the three colors are
    not mixed into one another.
    """
    kernel = _odd_kernel(size)
    if kernel <= 1:
        return rgb
    # scipy.ndimage.median_filter is the numerical engine; the definition
    # above is what it computes. Reflect padding avoids a dark frame at
    # the image border, which would otherwise look like a shoreline.
    filtered = median_filter(rgb, size=(kernel, kernel, 1), mode="reflect")
    return filtered


def apply_gaussian(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve each channel with an isotropic Gaussian of deviation σ.

    The continuous kernel is

        G_σ(x, y) = (1 / (2 π σ²)) exp(−(x² + y²) / (2 σ²))

    SciPy samples this kernel (truncated at about 4σ) and normalizes it so
    the weights sum to 1. A constant image is therefore unchanged, and the
    filter is a low-pass: a sinusoid of frequency f is multiplied by the
    Gaussian transform exp(−2 π² σ² f²), which decays as f grows.

    Sigma is in pixels. σ ≤ 0 leaves the image untouched. The channel axis
    has σ = 0 so color planes are filtered independently.
    """
    sigma = float(sigma)
    if sigma < 0.05:
        return rgb
    image = np.asarray(rgb, dtype=np.float32)
    blurred = gaussian_filter(image, sigma=(sigma, sigma, 0.0), mode="reflect")
    return blurred


def percentile_stretch(rgb: np.ndarray, low: float, high: float) -> np.ndarray:
    """Linear contrast stretch between two percentiles, per channel.

    For a channel C let p_low and p_high be the requested percentiles
    (defaults near 1 and 99, so a few saturated pixels do not set the
    scale). The displayed value is

        C' = clip( (C − p_low) / (p_high − p_low) , 0, 1 )

    This is an affine map. It does not invent structure; it spends the
    available output range on the intensities that actually occur.
    Channels are stretched separately because a scene can be globally
    dark in blue (deep water) and bright in red (soil).
    """
    low = float(np.clip(low, 0.0, 40.0))
    high = float(np.clip(high, 60.0, 100.0))
    if high <= low + 0.5:
        raise ValueError("The high percentile must be greater than the low percentile.")
    image = as_float01(rgb)
    out = np.empty_like(image)
    for channel in range(3):
        plane = image[..., channel]
        p_low = np.percentile(plane, low)
        p_high = np.percentile(plane, high)
        scale = float(p_high - p_low)
        if scale < 1e-6:
            out[..., channel] = plane
        else:
            out[..., channel] = np.clip((plane - p_low) / scale, 0.0, 1.0)
    return out


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Pointwise power-law map C' = C^γ on intensities in [0, 1].

    γ = 1 is the identity. γ < 1 lifts shadows (the derivative γ C^(γ−1)
    is large near 0). γ > 1 darkens them. Mid-tones move more than the
    endpoints 0 and 1, which stay fixed. The same γ is used on every
    channel so hue is approximately preserved.
    """
    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-3:
        return as_float01(rgb)
    if gamma <= 0.05 or gamma > 5.0:
        raise ValueError("Gamma must lie between 0.05 and 5.")
    image = np.clip(as_float01(rgb), 0.0, 1.0)
    return np.power(image, np.float32(gamma), dtype=np.float32)


def equalize_luminance(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize luma and scale RGB so hue is kept.

    Let g be the BT.601 luma and h(k) its 256-bin histogram. The
    cumulative distribution

        cdf(k) = Σ_{i=0..k} h(i) / N

    is a non-decreasing map from intensity to [0, 1]. Equalization sets

        g'(x, y) = cdf(g(x, y))

    which flattens the histogram when many pixels share few levels.
    Applying that map independently to R, G, and B would shift hue.
    Instead the color is scaled by the luma ratio

        (R', G', B') = (R, G, B) · g' / g

    with the ratio defined as 1 where g = 0. This is a simple von-Kries
    scaling: chromatic direction is unchanged, only the lightness moves.
    """
    image = as_float01(rgb)
    luma = rgb_to_gray(image)
    # 256 bins on [0, 1]. The last edge is 1, so clip before digitize.
    quantized = np.clip((luma * 255.0).astype(np.int32), 0, 255)
    histogram = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(histogram)
    cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0.0
    denom = cdf[-1] - cdf_min
    if denom <= 0:
        return image
    mapping = (cdf - cdf_min) / denom
    equalized = mapping[quantized].astype(np.float32)

    ratio = np.ones_like(luma, dtype=np.float32)
    valid = luma > 1e-4
    ratio[valid] = equalized[valid] / luma[valid]
    scaled = np.clip(image * ratio[..., None], 0.0, 1.0)
    return scaled.astype(np.float32, copy=False)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] or uint8 RGB array to contiguous uint8 RGB."""
    array = np.asarray(image)
    if array.dtype == np.uint8 and array.ndim == 3:
        return np.ascontiguousarray(array)
    return np.clip(np.round(as_float01(array) * 255.0), 0, 255).astype(np.uint8)


def preprocess(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Run the optional preprocessing chain and return uint8 RGB.

    Parameters are read from the shared workspace dictionary. Invalid
    combinations raise ValueError with a message the interface can show.
    """
    if rgb is None:
        raise ValueError("Load an image before preprocessing.")

    median_size = int(params.get("median_size", 1))
    sigma = float(params.get("gaussian_sigma", 0.0))
    mode = str(params.get("contrast_mode", "none"))

    identity = median_size <= 1 and sigma < 0.05 and mode == "none"
    if identity:
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    working = apply_median(rgb, median_size)
    working = apply_gaussian(working, sigma)

    if mode == "none":
        out = as_float01(working)
    elif mode == "stretch":
        out = percentile_stretch(working, float(params.get("stretch_low", 1)), float(params.get("stretch_high", 99)))
    elif mode == "gamma":
        out = apply_gamma(working, float(params.get("gamma", 1.0)))
    elif mode == "equalize":
        out = equalize_luminance(working)
    else:
        raise ValueError(f"Unknown contrast mode: {mode}")

    return to_uint8_rgb(out)


def luminance_histogram(rgb: np.ndarray, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of BT.601 luma, used by the preprocessing chart.

    Returns bin centers in [0, 1] and counts. The chart is a view of the
    same quantity the contrast tools modify; it is not a second definition
    of brightness.
    """
    luma = rgb_to_gray(rgb)
    counts, edges = np.histogram(luma, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float32), counts.astype(np.float32)
