"""Surface estimates and ground-truth scores.

Pixel accounting
----------------
Every pixel of the working image is a valid sample. This application does
not invent a no-data mask. If a photograph contains a black collar or an
overlay, those pixels are part of the image and are counted in the denominator.

    water percentage = (water pixels / total pixels) × 100

Physical area is a different statement. It requires a ground sampling
distance: the width and height, in meters, represented by one pixel.

    area_m2 = water pixels × pixel_width_m × pixel_height_m

If the user has not asserted that calibration, the area in square meters is
left undefined. A percentage is then the only honest summary. Reporting
"pixels × 1 m²" without a measured pixel size would present a unit the data
do not support.

Ground-truth scores
-------------------
For a predicted mask P and a reference mask G, with |·| the pixel count:

    TP = |P ∩ G|          FP = |P ∩ Gᶜ|
    FN = |Pᶜ ∩ G|         TN = |Pᶜ ∩ Gᶜ|

    precision = TP / (TP + FP)     fraction of the prediction that is water
    recall    = TP / (TP + FN)     fraction of the reference water found
    F1        = 2 PR / (P + R)     harmonic mean; equals the Dice coefficient
                                   for a single binary class
    IoU       = TP / (TP + FP + FN)
    accuracy  = (TP + TN) / N

When a denominator is zero because both masks are empty, the matching rate
is defined as 1: there was no water and none was predicted. Accuracy is
reported because it is requested, and the interface notes that it is
optimistic when water is a small part of the frame (the TN term dominates).
"""

from __future__ import annotations

import numpy as np

from watervision.processing.morphology import component_stats

METHOD_ORDER = ("hsv", "fourier", "hybrid")
METHOD_LABELS = {
    "hsv": "HSV",
    "fourier": "Fourier",
    "hybrid": "Hybrid",
}


def _as_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        # A saved RGB mask is accepted: any channel above mid-grey counts.
        array = array.max(axis=-1)
    if array.dtype == np.bool_:
        binary = array
    else:
        binary = array > (127 if np.issubdtype(array.dtype, np.integer) else 0.5)
    if binary.shape != shape:
        raise ValueError(
            f"Mask size {binary.shape[1]}×{binary.shape[0]} does not match the image "
            f"{shape[1]}×{shape[0]}."
        )
    return binary


def calibration_area(water_pixels: int, params: dict) -> float | None:
    """Square meters, or None when calibration is off or incomplete.

    Both pixel dimensions must be finite and strictly positive. The product
    is the area of one pixel only under the assumption of square-free
    (possibly rectangular) pixels and a locally constant scale. It is not a
    georeferenced measurement and it ignores terrain slope.
    """
    if not bool(params.get("use_calibration", False)):
        return None
    try:
        width = float(params.get("pixel_width_m", 0.0))
        height = float(params.get("pixel_height_m", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Pixel width and height must be numbers in meters.") from exc
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Pixel width and height must be positive when calibration is on.")
    return float(water_pixels) * width * height


def estimate_mask(mask: np.ndarray, params: dict) -> dict:
    """Pixel statistics and, when requested, a physical area for one mask."""
    binary = np.asarray(mask, dtype=bool)
    total = int(binary.size)
    water = int(binary.sum())
    percentage = (100.0 * water / total) if total else 0.0
    regions, largest = component_stats(binary)
    physical = calibration_area(water, params)
    if physical is None:
        note = (
            "Pixel-based estimate only. No spatial resolution is set, "
            "so this result is not a ground area in m²."
        )
    else:
        width = float(params["pixel_width_m"])
        height = float(params["pixel_height_m"])
        hectares = physical / 10000.0
        note = (
            f"Calibrated area using {width:.4g} m × {height:.4g} m pixels. "
            f"{physical:,.2f} m² ({hectares:,.4f} ha). "
            "This assumes a constant pixel scale and does not correct for slope."
        )
    return {
        "water_pixels": water,
        "total_pixels": total,
        "percentage": percentage,
        "regions": regions,
        "largest_region": largest,
        "physical_area_m2": physical,
        "note": note,
    }


def estimate_all(masks: dict[str, np.ndarray], params: dict) -> dict[str, dict]:
    """Estimate every method that is present. Missing methods are skipped."""
    results = {}
    for name in METHOD_ORDER:
        if name in masks and masks[name] is not None:
            results[name] = estimate_mask(masks[name], params)
    return results


def evaluate_mask(prediction: np.ndarray, reference: np.ndarray) -> dict:
    """IoU, precision, recall, F1, and pixel accuracy for one pair of masks."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(reference, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("The ground-truth mask must have the same size as the segmentation.")

    true_positive = int(np.count_nonzero(pred & truth))
    false_positive = int(np.count_nonzero(pred & ~truth))
    false_negative = int(np.count_nonzero(~pred & truth))
    true_negative = int(np.count_nonzero(~pred & ~truth))
    total = true_positive + false_positive + false_negative + true_negative

    def _rate(numerator: int, denominator: int) -> float:
        # Vacuous agreement: nothing was required and nothing was predicted.
        if denominator == 0:
            return 1.0
        return numerator / denominator

    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    iou = _rate(true_positive, true_positive + false_positive + false_negative)
    accuracy = _rate(true_positive + true_negative, total)
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


def evaluate_all(masks: dict[str, np.ndarray], reference: np.ndarray) -> dict[str, dict]:
    """Score each available method against the same reference mask."""
    return {
        name: evaluate_mask(masks[name], reference)
        for name in METHOD_ORDER
        if name in masks and masks[name] is not None
    }


def agreement_image(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Color-coded comparison used by the evaluation preview.

    True positive  — cyan, water found
    False positive — amber, predicted water that is dry in the reference
    False negative — violet, reference water that was missed
    True negative  — dark, correctly dry

    The colors are a display of the four sets in the definitions above.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(reference, dtype=bool)
    image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    image[..., :] = (12, 20, 32)
    image[pred & truth] = (34, 211, 238)
    image[pred & ~truth] = (245, 158, 11)
    image[~pred & truth] = (167, 139, 250)
    return image


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a binary mask to (height, width).

    Nearest neighbor is the interpolation that does not invent grey levels
    in a label image. A mask value must stay true or false. Linear
    interpolation would create fractions with no meaning as a class label.
    """
    binary = np.asarray(mask, dtype=bool)
    src_h, src_w = binary.shape
    dst_h, dst_w = shape
    if (src_h, src_w) == (dst_h, dst_w):
        return binary
    row = np.clip(np.round(np.linspace(0, src_h - 1, dst_h)).astype(np.int32), 0, src_h - 1)
    col = np.clip(np.round(np.linspace(0, src_w - 1, dst_w)).astype(np.int32), 0, src_w - 1)
    return binary[row][:, col]
