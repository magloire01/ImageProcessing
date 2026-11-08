"""Two-dimensional Fourier analysis.

The discrete Fourier transform used here is

    F(u, v) = Σ_x Σ_y  f(x, y) exp(−j 2π (u x / M + v y / N))

with the inverse

    f(x, y) = (1 / (M N)) Σ_u Σ_v  F(u, v) exp(+j 2π (u x / M + v y / N))

NumPy's fft2 / ifft2 evaluate these sums (the 1/(MN) factor lives in ifft2).
Before display or filtering, fftshift swaps quadrants so the zero frequency,
the DC component F(0, 0) = Σ f, sits at the center of the array. After that
swap, distance from the center is spatial frequency: the middle is smooth
structure, the corners are the finest oscillation the sampling grid can hold
(the Nyquist frequency).

Filtering multiplies F by a real transfer function H(u, v) and returns to
the spatial domain with the inverse transform. Multiplication in frequency
is convolution in space. A low-pass H blurs; a high-pass H removes the mean
and the slow shading, leaving edges and texture.

Local high-frequency energy is a different question. It does not filter the
whole image. It asks, inside a small window, what fraction of the window's
energy lives at high frequency. Calm water tends to score low. So do sand,
roads, and shadows. The number is a texture descriptor, not a water label.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import zoom

from watervision.processing.color import gray_to_rgb, rgb_to_gray


def fft2_shifted(gray: np.ndarray) -> np.ndarray:
    """Unitary-layout 2D DFT, centered, as complex64.

    The input is a real luminance image in [0, 1]. The output is fftshift(
    fft2(f)), so index (M//2, N//2) is the DC term. complex64 is enough for
    image-valued data and halves the memory of complex128.
    """
    field = np.ascontiguousarray(gray, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("The Fourier transform expects a single-channel image.")
    spectrum = np.fft.fft2(field)
    return np.fft.fftshift(spectrum).astype(np.complex64, copy=False)


def reconstruct(shifted: np.ndarray) -> np.ndarray:
    """Inverse of fft2_shifted, returning the real part.

    A real-valued image has a conjugate-symmetric spectrum. After a real
    filter H the symmetry is preserved, so the imaginary part of the inverse
    is numerical noise and is discarded. The real part is the reconstructed
    luminance, on the same scale as the input (approximately [0, 1] for a
    low-pass, approximately zero-mean for a high-pass).
    """
    unshifted = np.fft.ifftshift(shifted)
    spatial = np.fft.ifft2(unshifted)
    return np.real(spatial).astype(np.float32, copy=False)


def frequency_radius(shape: tuple[int, int]) -> np.ndarray:
    """Euclidean distance of every bin to the DC bin, in cycles/image.

    After fftshift the DC bin is at (rows // 2, cols // 2), matching NumPy's
    definition for both even and odd lengths. The radius is

        D(u, v) = sqrt( (u − u_c)² + (v − v_c)² )

    D = 0 at the mean of the image. D is largest at the corners.
    """
    rows, cols = shape
    center_y = rows // 2
    center_x = cols // 2
    y = np.arange(rows, dtype=np.float32) - np.float32(center_y)
    x = np.arange(cols, dtype=np.float32) - np.float32(center_x)
    return np.hypot(y[:, None], x[None, :])


def _cutoff_radius(shape: tuple[int, int], cutoff_fraction: float) -> float:
    """Map a unitless fraction in (0, 1] to a radius in frequency bins.

    The fraction is relative to the corner of the centered spectrum, which
    is the highest frequency present along the diagonal. A fraction of 1
    therefore admits (or, for a high-pass, starts from) the entire spectrum.
    Using a fraction rather than a raw bin count keeps the slider meaningful
    when the image size changes.
    """
    rows, cols = shape
    corner = float(np.hypot(rows // 2, cols // 2))
    fraction = float(np.clip(cutoff_fraction, 1e-3, 1.0))
    return max(fraction * max(corner, 1.0), 1e-3)


def butterworth_lowpass(shape: tuple[int, int], cutoff_fraction: float, order: int) -> np.ndarray:
    """Radially symmetric Butterworth low-pass of order n.

        H_lp(u, v) = 1 / ( 1 + (D(u, v) / D0) ^ (2 n) )

    H(0) = 1, so the mean of the image is kept. H(D0) = 1/2 for every order:
    D0 is the half-power radius. Larger n makes the transition from pass-band
    to stop-band steeper and closer to an ideal disk. There is no hard edge,
    so the spatial kernel decays instead of ringing as strongly as a sinc.
    """
    order = int(np.clip(order, 1, 8))
    radius = frequency_radius(shape)
    d0 = _cutoff_radius(shape, cutoff_fraction)
    ratio = radius / np.float32(d0)
    # (2n) is even for integer n, so the ratio is safely non-negative.
    return (1.0 / (1.0 + ratio ** (2 * order))).astype(np.float32)


def butterworth_highpass(shape: tuple[int, int], cutoff_fraction: float, order: int) -> np.ndarray:
    """Classical Butterworth high-pass.

        H_hp(u, v) = 1 / ( 1 + (D0 / D(u, v)) ^ (2 n) )    for D > 0
        H_hp(0, 0) = 0

    This is the standard complement used in textbooks. It is not identical
    to 1 − H_lp, but it has the same half-power radius and the same limits
    (0 at DC, 1 at infinity). The center sample is written explicitly because
    D0 / D is undefined at the origin.
    """
    order = int(np.clip(order, 1, 8))
    radius = frequency_radius(shape)
    d0 = _cutoff_radius(shape, cutoff_fraction)
    transfer = np.zeros(shape, dtype=np.float32)
    nonzero = radius > 0
    ratio = np.float32(d0) / radius[nonzero]
    transfer[nonzero] = 1.0 / (1.0 + ratio ** (2 * order))
    return transfer


def ideal_lowpass(shape: tuple[int, int], cutoff_fraction: float) -> np.ndarray:
    """Ideal disk low-pass: H = 1 inside the cutoff radius, else 0.

    The inverse transform of a disk is a jinc (a Bessel beam). Convolution
    with that kernel produces Gibbs ringing around strong edges. The filter
    is available because it is the definition students meet first; the
    Butterworth is the better default for a photograph.
    """
    radius = frequency_radius(shape)
    d0 = _cutoff_radius(shape, cutoff_fraction)
    return (radius <= d0).astype(np.float32)


def ideal_highpass(shape: tuple[int, int], cutoff_fraction: float) -> np.ndarray:
    """Ideal complement of the disk low-pass, including a zero at DC."""
    return (1.0 - ideal_lowpass(shape, cutoff_fraction)).astype(np.float32)


def transfer_function(shape: tuple[int, int], params: dict) -> np.ndarray:
    """Build H(u, v) from the workspace filter settings."""
    kind = str(params.get("filter_kind", "low-pass"))
    response = str(params.get("filter_response", "butterworth"))
    cutoff = float(params.get("cutoff", 0.2))
    order = int(params.get("butterworth_order", 2))
    if response not in ("ideal", "butterworth"):
        raise ValueError("Filter response must be 'ideal' or 'butterworth'.")
    if kind == "low-pass" and response == "ideal":
        return ideal_lowpass(shape, cutoff)
    if kind == "low-pass":
        return butterworth_lowpass(shape, cutoff, order)
    if kind == "high-pass" and response == "ideal":
        return ideal_highpass(shape, cutoff)
    if kind == "high-pass":
        return butterworth_highpass(shape, cutoff, order)
    raise ValueError("Filter kind must be 'low-pass' or 'high-pass'.")


def apply_transfer(shifted: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Pointwise product G(u, v) = H(u, v) F(u, v).

    H is real and radial. Scaling a complex coefficient scales its magnitude
    and leaves its phase unchanged, so the filter does not shift edges, it
    only attenuates selected frequencies.
    """
    return shifted * transfer.astype(np.float32, copy=False)


def log_magnitude(shifted: np.ndarray) -> np.ndarray:
    """Logarithmic magnitude used only for visualization.

        M(u, v) = log(1 + |F(u, v)|)

    |F| = sqrt(Re² + Im²) spans several orders of magnitude because the DC
    term is the sum of every pixel. The logarithm compresses that range so
    the mid and high frequencies remain visible. The 1 inside the log keeps
    a zero coefficient at zero. This array is never transformed back; the
    inverse FFT always uses the complex spectrum.
    """
    magnitude = np.abs(shifted).astype(np.float32, copy=False)
    return np.log1p(magnitude)


def colorize_spectrum(shifted: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Render log|F| with a matplotlib colormap for the side-by-side view.

    A percentile stretch (1st to 99.8th) is applied after the logarithm so a
    single bright DC sample does not paint the rest of the spectrum black.
    The stretch is a display transform. It does not modify F.
    """
    field = log_magnitude(shifted)
    low, high = np.percentile(field, (1.0, 99.8))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((field - low) / (high - low), 0.0, 1.0)
    try:
        cmap = colormaps[cmap_name]
    except (KeyError, ValueError):
        cmap = colormaps["magma"]
    colored = cmap(normalized)[..., :3]
    return np.clip(np.round(colored * 255.0), 0, 255).astype(np.uint8)


def reconstruction_to_rgb(field: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    """Prepare a filtered luminance image for display.

    A low-pass result is a smoothed image on the original [0, 1] scale and
    is clipped. A high-pass result is a signed residual: the mean was
    removed, so half the pixels are negative and a direct clip to black
    would hide the texture. For that case the residual is shifted and
    scaled to [0, 1] for display only. The label records which of the two
    presentations the user is looking at.
    """
    image = np.asarray(field, dtype=np.float32)
    if kind == "high-pass":
        low = float(np.percentile(image, 1.0))
        high = float(np.percentile(image, 99.0))
        if high <= low:
            high = low + 1.0
        shown = np.clip((image - low) / (high - low), 0.0, 1.0)
        note = "High-pass residual, contrast-stretched for display. The mean was removed."
    else:
        shown = np.clip(image, 0.0, 1.0)
        note = "Low-pass reconstruction on the original luminance scale."
    return gray_to_rgb(shown), note


def analyze_spectrum(rgb: np.ndarray, params: dict) -> dict:
    """Compute the centered DFT, its display image, and the filtered image.

    The luminance transform is the one defined in color.rgb_to_gray. Phase
    is retained in the cached complex array so a later change of H does not
    require another FFT.
    """
    gray = rgb_to_gray(rgb)
    shifted = fft2_shifted(gray)
    cmap_name = str(params.get("spectrum_cmap", "magma"))
    spectrum_rgb = colorize_spectrum(shifted, cmap_name)
    transfer = transfer_function(shifted.shape, params)
    filtered = reconstruct(apply_transfer(shifted, transfer))
    kind = str(params.get("filter_kind", "low-pass"))
    filtered_rgb, note = reconstruction_to_rgb(filtered, kind)
    return {
        "spectrum": shifted,
        "spectrum_rgb": spectrum_rgb,
        "filtered_rgb": filtered_rgb,
        "filter_label": note,
    }


def filter_cached(shifted: np.ndarray, params: dict) -> dict:
    """Recompute H and the inverse FFT from a cached spectrum."""
    transfer = transfer_function(shifted.shape, params)
    filtered = reconstruct(apply_transfer(shifted, transfer))
    kind = str(params.get("filter_kind", "low-pass"))
    filtered_rgb, note = reconstruction_to_rgb(filtered, kind)
    spectrum_rgb = colorize_spectrum(shifted, str(params.get("spectrum_cmap", "magma")))
    return {
        "spectrum_rgb": spectrum_rgb,
        "filtered_rgb": filtered_rgb,
        "filter_label": note,
    }


def _hann(window: int) -> np.ndarray:
    """Separable 2D Hann window.

    w(n) = 0.5 − 0.5 cos(2π n / (N − 1)), n = 0 … N − 1, and W = w wᵀ.

    A rectangular patch has a sharp boundary. The DFT treats the patch as
    one period of a periodic signal, so that jump leaks energy into high
    frequencies that are not in the texture (spectral leakage). Multiplying
    by W tapers the patch to zero at its border and reduces the leak. The
    same window is used for every patch, so the energy ratio stays comparable.
    """
    if window <= 1:
        return np.ones((window, window), dtype=np.float32)
    n = np.arange(window, dtype=np.float32)
    tap = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / np.float32(window - 1))
    return np.outer(tap, tap)


def local_high_frequency_ratio(
    gray: np.ndarray,
    window: int = 32,
    cutoff_ratio: float = 0.35,
    max_windows: int = 56,
) -> np.ndarray:
    """High-frequency energy ratio on a grid of windowed patches.

    For a patch p, after the mean is removed and the Hann window is applied,
    let P = fftshift(fft2(p)) and let |P|² be the energy spectral density.
    Parseval's relation says the sum of |P|² equals N² times the energy of
    the windowed patch, so ratios of sums of |P|² are ratios of energy.

    Split the bins into a low-frequency disk of radius

        D0 = cutoff_ratio · (window / 2)

    and its complement. The reported ratio is

        E_high / E_total = Σ_{D > D0} |P|²  /  Σ |P|²

    A perfectly flat patch has E_total ≈ 0 after the mean is removed; its
    ratio is defined as 0 (no high-frequency energy). A noisy or vegetated
    patch has a large ratio.

    Patches are taken on a regular stride chosen so each axis contributes at
    most `max_windows` spectra. The coarse ratio image is then bilinearly
    resampled to the full raster. The resampling does not create new spectral
    measurements; it only lets later pixel-wise rules read a value.

    This map characterizes texture. It is not a water detector. Any smooth
    surface — sand, a roof, still shadow — produces a low ratio as well.
    """
    field = np.ascontiguousarray(np.asarray(gray, dtype=np.float32))
    if field.ndim != 2:
        raise ValueError("Local frequency analysis expects a luminance image.")
    height, width = field.shape
    window = int(np.clip(window, 8, 64))
    cutoff_ratio = float(np.clip(cutoff_ratio, 0.05, 0.95))
    max_windows = int(np.clip(max_windows, 8, 80))

    if height < window or width < window:
        # The whole image is one patch. The result is a constant field.
        ratio = _patch_ratios(field[None, None, ...], window, cutoff_ratio)
        return np.full((height, width), float(ratio.reshape(-1)[0]), dtype=np.float32)

    stride_y = max(1, int(np.ceil((height - window + 1) / max_windows)))
    stride_x = max(1, int(np.ceil((width - window + 1) / max_windows)))
    view = sliding_window_view(field, (window, window))
    patches = np.ascontiguousarray(view[::stride_y, ::stride_x])
    ratios = _patch_ratios(patches, window, cutoff_ratio)

    # zoom factors map the coarse grid onto the full image. Crop or pad by a
    # pixel if rounding does not land exactly on (height, width).
    zoom_y = height / ratios.shape[0]
    zoom_x = width / ratios.shape[1]
    upsampled = zoom(ratios, (zoom_y, zoom_x), order=1)
    upsampled = np.asarray(upsampled, dtype=np.float32)
    if upsampled.shape[0] < height or upsampled.shape[1] < width:
        padded = np.pad(
            upsampled,
            ((0, max(0, height - upsampled.shape[0])), (0, max(0, width - upsampled.shape[1]))),
            mode="edge",
        )
        upsampled = padded
    return upsampled[:height, :width]


def _patch_ratios(patches: np.ndarray, window: int, cutoff_ratio: float) -> np.ndarray:
    """Energy ratio for a stack of patches with shape (..., window, window)."""
    centered = patches - patches.mean(axis=(-2, -1), keepdims=True)
    windowed = centered * _hann(window)
    spectrum = np.fft.fftshift(np.fft.fft2(windowed, axes=(-2, -1)), axes=(-2, -1))
    power = spectrum.real.astype(np.float32) ** 2 + spectrum.imag.astype(np.float32) ** 2

    cy = window // 2
    cx = window // 2
    y = np.arange(window, dtype=np.float32) - np.float32(cy)
    x = np.arange(window, dtype=np.float32) - np.float32(cx)
    radius = np.hypot(y[:, None], x[None, :])
    high = radius > (cutoff_ratio * (window / 2.0))

    high_energy = (power * high).sum(axis=(-2, -1))
    total_energy = power.sum(axis=(-2, -1))
    ratio = np.zeros(total_energy.shape, dtype=np.float32)
    np.divide(high_energy, total_energy, out=ratio, where=total_energy > 1e-8)
    return ratio.astype(np.float32, copy=False)


def energy_to_rgb(energy: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Color the local high-frequency ratio for the Fourier page.

    Bright colors mean textured patches. Dark colors mean smooth patches.
    The scale is the actual ratio in [0, 1], not a percentile stretch, so
    two images can be compared visually.
    """
    field = np.clip(np.asarray(energy, dtype=np.float32), 0.0, 1.0)
    try:
        cmap = colormaps[cmap_name]
    except (KeyError, ValueError):
        cmap = colormaps["magma"]
    colored = cmap(field)[..., :3]
    return np.clip(np.round(colored * 255.0), 0, 255).astype(np.uint8)
