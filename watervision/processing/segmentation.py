"""Three water-segmentation methods that share one preprocessing image.

Method 1 — HSV threshold
    A pixel is water when its hue, saturation, and value fall inside a
    user interval. A second hue interval can be enabled for turbid brown
    water, and a dark-water rule can accept low-saturation pixels that have
    no reliable hue. The test is Boolean. There is no texture term.

Method 2 — local frequency
    A pixel is a *candidate* when its high-frequency energy ratio is below
    a threshold, i.e. when the surrounding patch is smooth. Smoothness is
    common to calm water and to many other surfaces. This mask must not be
    read as detected water. It is an intermediate texture mask, kept so the
    contribution of frequency can be seen on its own.

Method 3 — hybrid score (the method intended for the area estimate)
    Each pixel receives a score in [0, 1]

        S = w_c C + w_t T + w_h H
        w_c + w_t + w_h = 1,   weights ≥ 0

    C is a soft HSV membership (the same bands as method 1, with a gradual
    falloff outside the intervals, so a shoreline is not a hard color cliff).
    T is a logistic soft version of the method-2 threshold: smooth patches
    score near 1, textured patches near 0.
    H is local homogeneity, 1 when the local standard deviation is far below
    a reference and 0 when it is far above. Homogeneity is optional.

    The pixel is labeled water when S ≥ t. Because C is required to carry
    a large share of S unless the user raises the texture weight, a smooth
    tan roof does not become water on frequency evidence alone.

The default interval keeps blue pixels only, from 185° to 255° (cyan-blue
through pure blue). Green and yellow fall outside that arc. The ochre band
and the dark-water rule stay in the code, but they are off unless the user
turns them on. All three gates remain adjustable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from watervision.processing.color import rgb_to_gray, rgb_to_hsv
from watervision.processing.fourier import local_high_frequency_ratio
from watervision.processing.morphology import apply_morphology, component_stats


def hue_in_range(hue: np.ndarray, low: float, high: float) -> np.ndarray:
    """Circular interval test on hue in degrees.

    If low ≤ high the interval is an ordinary arc. If low > high the arc
    crosses 0° (reds), and a hue is inside when it is ≥ low or ≤ high.
    """
    low = float(low) % 360.0
    high = float(high) % 360.0
    if low <= high:
        return (hue >= low) & (hue <= high)
    return (hue >= low) | (hue <= high)


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute difference on a 360° circle."""
    delta = np.abs(a - np.float32(b)) % np.float32(360.0)
    return np.minimum(delta, np.float32(360.0) - delta)


def hue_interval_distance(hue: np.ndarray, low: float, high: float) -> np.ndarray:
    """Angular distance from each hue to the nearest point of the interval.

    Inside the interval the distance is 0. Outside, the nearest point of a
    closed arc is one of the two endpoints, so the distance is the minimum
    of the two circular endpoint distances.
    """
    inside = hue_in_range(hue, low, high)
    distance = np.minimum(_circular_distance(hue, low), _circular_distance(hue, high))
    return np.where(inside, np.float32(0.0), distance).astype(np.float32)


def _interval_score(values: np.ndarray, low: float, high: float, softness: float) -> np.ndarray:
    """1 inside [low, high], Gaussian decay outside. Softness is the σ."""
    if high < low:
        low, high = high, low
    below = np.maximum(low - values, 0.0)
    above = np.maximum(values - high, 0.0)
    distance = np.maximum(below, above)
    sigma = max(float(softness), 1e-3)
    return np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)


def _band_score(
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    h_low: float,
    h_high: float,
    s_low: float,
    s_high: float,
    v_low: float,
    v_high: float,
    hue_softness: float,
) -> np.ndarray:
    """Soft membership of one HSV box.

    All three gates multiply:

        C = C_H · C_S · C_V

    A bright tan surface can sit near a brown hue and still be dry. If value
    were only allowed to weaken the score, that surface would keep a large
    color term and the texture term would push it over the decision threshold.
    The product forces hue, saturation, and value to agree. At the default
    weights the texture and homogeneity terms sum to less than the decision
    threshold, so smoothness alone cannot label a pixel as water.
    """
    hue_score = np.exp(
        -0.5 * (hue_interval_distance(hue, h_low, h_high) / max(hue_softness, 1.0)) ** 2
    ).astype(np.float32)
    sat_score = _interval_score(sat, s_low, s_high, 0.06)
    val_score = _interval_score(val, v_low, v_high, 0.06)
    return hue_score * sat_score * val_score


def dark_water_score(sat: np.ndarray, val: np.ndarray, v_max: float, s_max: float) -> np.ndarray:
    """Membership of the dark, weakly saturated class.

    Hue is unstable when saturation is near zero, which is exactly the case
    for grey or black water. The score is high only when value is at most
    v_max and saturation is at most s_max, with a Gaussian tail past those
    limits. Bright grey pavement fails the value test. Saturated vegetation
    fails the saturation test.
    """
    v_excess = np.maximum(val - np.float32(v_max), 0.0)
    s_excess = np.maximum(sat - np.float32(s_max), 0.0)
    v_term = np.exp(-0.5 * (v_excess / np.float32(0.06)) ** 2)
    s_term = np.exp(-0.5 * (s_excess / np.float32(0.08)) ** 2)
    return (v_term * s_term).astype(np.float32)


def color_membership(hsv: np.ndarray, params: dict) -> np.ndarray:
    """Soft color score in [0, 1], the C term of the hybrid method."""
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    softness = float(params.get("hue_softness", 16.0))
    primary = _band_score(
        hue, sat, val,
        float(params["h_min"]), float(params["h_max"]),
        float(params["s_min"]), float(params["s_max"]),
        float(params["v_min"]), float(params["v_max"]),
        softness,
    )
    score = primary
    if bool(params.get("use_secondary_hue", False)):
        secondary = _band_score(
            hue, sat, val,
            float(params["h2_min"]), float(params["h2_max"]),
            float(params["s2_min"]), float(params["s2_max"]),
            float(params["v2_min"]), float(params["v2_max"]),
            softness,
        )
        score = np.maximum(score, secondary)
    if bool(params.get("include_dark_water", False)):
        dark = dark_water_score(
            sat, val, float(params["dark_v_max"]), float(params["dark_s_max"])
        )
        score = np.maximum(score, dark)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def hsv_hard_mask(hsv: np.ndarray, params: dict) -> np.ndarray:
    """Method 1. Boolean HSV box, optional second hue, optional dark rule.

    A pixel is accepted when it lies in the primary box, or in the secondary
    box if that band is enabled, or in the dark-water box if that rule is
    enabled. Edges of the boxes are sharp: this is the classical threshold
    method, and it is deliberately not the soft score used by the hybrid.
    """
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    def _box(h0, h1, s0, s1, v0, v1) -> np.ndarray:
        return (
            hue_in_range(hue, h0, h1)
            & (sat >= s0) & (sat <= s1)
            & (val >= v0) & (val <= v1)
        )

    mask = _box(
        float(params["h_min"]), float(params["h_max"]),
        float(params["s_min"]), float(params["s_max"]),
        float(params["v_min"]), float(params["v_max"]),
    )
    if bool(params.get("use_secondary_hue", False)):
        mask = mask | _box(
            float(params["h2_min"]), float(params["h2_max"]),
            float(params["s2_min"]), float(params["s2_max"]),
            float(params["v2_min"]), float(params["v2_max"]),
        )
    if bool(params.get("include_dark_water", False)):
        dark = (val <= float(params["dark_v_max"])) & (sat <= float(params["dark_s_max"]))
        mask = mask | dark
    return mask


def logistic_smoothness(energy: np.ndarray, threshold: float, steepness: float) -> np.ndarray:
    """Soft texture score. Smooth (low energy) maps toward 1.

        T(E) = 1 / (1 + exp(k (E − t)))

    t is the same threshold the hard frequency mask uses. k controls how
    quickly the score falls once a patch becomes textured. The argument of
    the exponential is clipped so a very large k cannot overflow.
    """
    z = np.clip(float(steepness) * (energy - float(threshold)), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def local_homogeneity(gray: np.ndarray, window: int, std_ref: float) -> np.ndarray:
    """Map local standard deviation to a homogeneity score in [0, 1].

    The local variance inside a box of side `window` is estimated by

        Var = E[X²] − (E[X])²

    with both expectations replaced by a normalized box filter (uniform_filter).
    The standard deviation σ = sqrt(max(Var, 0)) is then compared with a
    reference σ_ref (a fraction of the [0, 1] luma scale):

        H = clip(1 − σ / σ_ref, 0, 1)

    H = 1 for a flat patch and H = 0 once the window is at least as rough as
    the reference. This is not the same statement as the Fourier ratio.
    A slow shade across a lake has variance but almost no high-frequency
    energy. Ripples and foliage have both. Keeping both terms lets the user
    decide which description of "smooth" matters.
    """
    size = int(window)
    if size % 2 == 0:
        size += 1
    size = int(np.clip(size, 3, 65))
    field = np.asarray(gray, dtype=np.float32)
    mean = uniform_filter(field, size=size, mode="reflect")
    mean_sq = uniform_filter(field * field, size=size, mode="reflect")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)
    reference = max(float(std_ref), 1e-4)
    return np.clip(1.0 - std / reference, 0.0, 1.0).astype(np.float32)


def normalized_weights(params: dict) -> dict[str, float]:
    """Return color, texture, and homogeneity weights that sum to 1.

    A zero sum (the user slid every weight to 0, or homogeneity is off and
    the other two are 0) falls back to color only, so the score stays defined.
    """
    color = max(float(params.get("weight_color", 0.0)), 0.0)
    texture = max(float(params.get("weight_texture", 0.0)), 0.0)
    hom = max(float(params.get("weight_homogeneity", 0.0)), 0.0)
    if not bool(params.get("use_homogeneity", True)):
        hom = 0.0
    total = color + texture + hom
    if total <= 1e-8:
        return {"color": 1.0, "texture": 0.0, "homogeneity": 0.0}
    return {"color": color / total, "texture": texture / total, "homogeneity": hom / total}


def frequency_candidate_mask(energy: np.ndarray, threshold: float) -> np.ndarray:
    """Method 2. True where the local spectrum is smoother than `threshold`.

    The name is deliberate: the result is a candidate mask. It is displayed
    and scored so the frequency cue can be inspected. It is not, by itself,
    a claim that water is present.
    """
    return np.asarray(energy, dtype=np.float32) < float(threshold)


@dataclass
class SegmentationResult:
    """Everything the interface needs after one segmentation run."""

    energy: np.ndarray
    raw: dict[str, np.ndarray]
    clean: dict[str, np.ndarray]
    weights: dict[str, float]
    # Hybrid score, kept so a report can state the decision threshold against it.
    score: np.ndarray


def segment(rgb: np.ndarray, params: dict, energy: np.ndarray | None = None) -> SegmentationResult:
    """Run all three methods and the shared morphological cleanup.

    `energy` may be supplied when the Fourier page has already computed it
    for the same image and the same patch settings. Otherwise it is computed
    here. The three raw masks and the three cleaned masks are returned.
    Metrics are intentionally not computed here; the metrics module owns the
    definitions of percentage, area, and region size.
    """
    if rgb is None or rgb.ndim != 3:
        raise ValueError("Segmentation requires an RGB working image.")

    hsv = rgb_to_hsv(rgb)
    gray = rgb_to_gray(rgb)

    window = int(params.get("patch_window", 32))
    cutoff = float(params.get("patch_cutoff", 0.35))
    max_windows = int(params.get("max_frequency_windows", 56))
    if energy is None or energy.shape[:2] != gray.shape:
        energy = local_high_frequency_ratio(gray, window, cutoff, max_windows)
    else:
        energy = np.asarray(energy, dtype=np.float32)

    raw_hsv = hsv_hard_mask(hsv, params)
    raw_frequency = frequency_candidate_mask(energy, float(params.get("texture_threshold", 0.4)))

    color = color_membership(hsv, params)
    texture = logistic_smoothness(
        energy,
        float(params.get("texture_threshold", 0.4)),
        float(params.get("texture_steepness", 14.0)),
    )
    weights = normalized_weights(params)
    if weights["homogeneity"] > 0.0:
        hom = local_homogeneity(
            gray, window, float(params.get("homogeneity_std_ref", 0.08))
        )
    else:
        hom = np.zeros(gray.shape, dtype=np.float32)

    score = (
        np.float32(weights["color"]) * color
        + np.float32(weights["texture"]) * texture
        + np.float32(weights["homogeneity"]) * hom
    ).astype(np.float32)
    raw_hybrid = score >= float(params.get("decision_threshold", 0.58))

    raw = {"hsv": raw_hsv, "fourier": raw_frequency, "hybrid": raw_hybrid}
    clean = {name: apply_morphology(mask, params) for name, mask in raw.items()}
    # Closing and hole filling grow the mask. On a globe the ocean encloses
    # each continent, so those land pixels would be filled even though they
    # are not blue. The displayed water masks are therefore cut back to the
    # pixels that pass the color test. The Fourier mask stays a texture map.
    clean["hsv"] = clean["hsv"] & raw_hsv
    clean["hybrid"] = clean["hybrid"] & raw_hsv
    return SegmentationResult(energy=energy, raw=raw, clean=clean, weights=weights, score=score)


def region_count(mask: np.ndarray) -> tuple[int, int]:
    """Convenience wrapper so callers do not import morphology for one query."""
    return component_stats(mask)
