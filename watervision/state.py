"""Shared workspace state for one analysis session.

The interface and the numerical pipeline meet here. Image arrays stay in
this object so pages do not each keep a private copy of a large raster.
Downstream products (spectrum, masks, metrics) are dropped whenever the
working image changes, because they would otherwise describe an image the
user can no longer see.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

# Identity preprocessing: the working image is the file the user loaded.
# Recommended analysis values are documented in the interface, not forced
# onto the data before the user asks for them.
DEFAULT_PARAMS: dict[str, object] = {
    # --- preprocessing -------------------------------------------------
    "median_size": 1,
    "gaussian_sigma": 0.0,
    "contrast_mode": "none",  # none | stretch | gamma | equalize
    "stretch_low": 1.0,
    "stretch_high": 99.0,
    "gamma": 1.0,
    # --- full-image Fourier filter (visualization) --------------------
    "filter_kind": "low-pass",  # low-pass | high-pass
    "filter_response": "butterworth",  # ideal | butterworth
    "cutoff": 0.18,
    "butterworth_order": 2,
    "spectrum_cmap": "magma",
    # --- local frequency description ----------------------------------
    # patch_cutoff is the fraction of the patch Nyquist radius. Energy
    # outside that disk is counted as high frequency.
    "patch_window": 32,
    "patch_cutoff": 0.35,
    "max_frequency_windows": 56,
    # --- method 1: hard HSV thresholds --------------------------------
    "h_min": 90.0,
    "h_max": 255.0,
    "s_min": 0.12,
    "s_max": 1.0,
    "v_min": 0.08,
    "v_max": 0.92,
    "use_secondary_hue": True,
    "h2_min": 8.0,
    "h2_max": 42.0,
    "s2_min": 0.10,
    "s2_max": 0.65,
    "v2_min": 0.08,
    "v2_max": 0.55,
    "include_dark_water": True,
    "dark_v_max": 0.20,
    "dark_s_max": 0.35,
    # --- method 2: smooth-region candidate from local HF energy -------
    "texture_threshold": 0.42,
    "texture_steepness": 14.0,
    # --- method 3: hybrid score ----------------------------------------
    "weight_color": 0.50,
    "weight_texture": 0.35,
    "weight_homogeneity": 0.15,
    "use_homogeneity": True,
    "decision_threshold": 0.56,
    "hue_softness": 16.0,
    "homogeneity_std_ref": 0.08,
    # --- morphology ----------------------------------------------------
    "morph_radius": 2,
    "do_opening": True,
    "do_closing": True,
    "do_fill_holes": True,
    "do_remove_small": True,
    "min_region_pixels": 80,
    # --- which cleaned mask feeds area estimation and export ----------
    "active_method": "hybrid",
    # --- ground sampling distance; ignored unless calibration is on ---
    "use_calibration": False,
    "pixel_width_m": 1.0,
    "pixel_height_m": 1.0,
}

PREPROCESS_KEYS = (
    "median_size",
    "gaussian_sigma",
    "contrast_mode",
    "stretch_low",
    "stretch_high",
    "gamma",
)

# Parameters owned by the segmentation page. Resetting them does not touch
# preprocessing, calibration, or the full-image Fourier display settings.
SEGMENT_KEYS = (
    "patch_window",
    "patch_cutoff",
    "h_min",
    "h_max",
    "s_min",
    "s_max",
    "v_min",
    "v_max",
    "use_secondary_hue",
    "h2_min",
    "h2_max",
    "s2_min",
    "s2_max",
    "v2_min",
    "v2_max",
    "include_dark_water",
    "dark_v_max",
    "dark_s_max",
    "texture_threshold",
    "texture_steepness",
    "weight_color",
    "weight_texture",
    "weight_homogeneity",
    "use_homogeneity",
    "decision_threshold",
    "hue_softness",
    "homogeneity_std_ref",
    "morph_radius",
    "do_opening",
    "do_closing",
    "do_fill_holes",
    "do_remove_small",
    "min_region_pixels",
    "active_method",
)


def default_params() -> dict[str, object]:
    return deepcopy(DEFAULT_PARAMS)


@dataclass
class AppState:
    """Arrays and numbers that describe the current workspace."""

    original: np.ndarray | None = None
    working: np.ndarray | None = None
    filename: str = ""
    path: str = ""
    load_note: str = ""

    # Complex spectrum is cached so cutoff changes do not recompute the FFT.
    # Stored after fftshift, as complex64, to limit memory.
    spectrum: np.ndarray | None = None
    spectrum_revision: int = -1
    spectrum_rgb: np.ndarray | None = None
    filtered_rgb: np.ndarray | None = None
    filter_label: str = ""

    energy: np.ndarray | None = None
    energy_key: tuple | None = None

    mask_raw: dict[str, np.ndarray] = field(default_factory=dict)
    mask_clean: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)

    ground_truth: np.ndarray | None = None
    ground_truth_name: str = ""
    ground_truth_resized: bool = False
    evaluation: dict[str, dict] = field(default_factory=dict)

    params: dict[str, object] = field(default_factory=default_params)
    applied_preprocess: dict[str, object] = field(default_factory=dict)

    revision: int = 0
    status: str = "Ready"
    last_timing_s: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def has_image(self) -> bool:
        return self.working is not None

    @property
    def height(self) -> int:
        return 0 if self.working is None else int(self.working.shape[0])

    @property
    def width(self) -> int:
        return 0 if self.working is None else int(self.working.shape[1])

    def set_original(self, rgb: np.ndarray, filename: str, path: str, note: str = "") -> None:
        """Replace the workspace with a newly acquired image."""
        self.original = rgb
        self.working = rgb.copy()
        self.filename = filename
        self.path = path
        self.load_note = note
        # A new acquisition is not a processed image. Mark preprocessing as
        # identity so the warning stays quiet until the user moves a control.
        self.applied_preprocess = {
            "median_size": 1,
            "gaussian_sigma": 0.0,
            "contrast_mode": "none",
            "stretch_low": float(self.params["stretch_low"]),
            "stretch_high": float(self.params["stretch_high"]),
            "gamma": 1.0,
        }
        self.ground_truth = None
        self.ground_truth_name = ""
        self.ground_truth_resized = False
        self.evaluation = {}
        self._invalidate_products()
        self.revision += 1
        self.status = "Image loaded"
        self.last_timing_s = None

    def set_working(self, rgb: np.ndarray, preprocess_params: dict[str, object]) -> None:
        """Commit a preprocessed image and retire stale measurements."""
        self.working = rgb
        self.applied_preprocess = dict(preprocess_params)
        self._invalidate_products()
        self.revision += 1
        self.status = "Preprocessing applied"

    def _invalidate_products(self) -> None:
        self.spectrum = None
        self.spectrum_revision = -1
        self.spectrum_rgb = None
        self.filtered_rgb = None
        self.filter_label = ""
        self.energy = None
        self.energy_key = None
        self.mask_raw = {}
        self.mask_clean = {}
        self.metrics = {}
        self.weights_used = {}
        self.evaluation = {}

    def reset_workspace(self) -> None:
        """Return to the empty state. Parameters return to their defaults."""
        self.original = None
        self.working = None
        self.filename = ""
        self.path = ""
        self.load_note = ""
        self.ground_truth = None
        self.ground_truth_name = ""
        self.ground_truth_resized = False
        self.evaluation = {}
        self.params = default_params()
        self.applied_preprocess = {}
        self._invalidate_products()
        self.revision += 1
        self.status = "Workspace reset"
        self.last_timing_s = None
        self.timings = {}

    def active_mask(self) -> np.ndarray | None:
        """Mask currently used for area estimation and the main overlay."""
        method = str(self.params.get("active_method", "hybrid"))
        cleaned = self.mask_clean.get(method)
        if cleaned is not None:
            return cleaned
        return self.mask_raw.get(method)

    def preprocess_is_dirty(self) -> bool:
        if not self.applied_preprocess:
            return False
        for key in PREPROCESS_KEYS:
            if self.params.get(key) != self.applied_preprocess.get(key):
                return True
        return False
