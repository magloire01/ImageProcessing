"""Spatial preprocessing of the working image.

The preview is computed from the original file plus the current sliders, in
a background thread. Nothing is written into the workspace until Apply.
Analysis modules read the applied image only, so a half-moved slider cannot
silently change a Fourier result.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.filters import luminance_histogram, preprocess
from watervision.state import DEFAULT_PARAMS, PREPROCESS_KEYS
from watervision.theme import COLORS, FONT_SMALL
from watervision.ui.charts import ChartPanel
from watervision.ui.widgets import Hint, ImageCard, SectionLabel, SliderRow, primary_button, secondary_button


class PreprocessingPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._preview = None
        self._preview_params: dict | None = None
        self._generation = 0
        self._job = None

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        stage = ctk.CTkFrame(self, fg_color="transparent")
        stage.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_columnconfigure(1, weight=1)
        stage.grid_rowconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=0)
        self.original_card = ImageCard(stage, "Original")
        self.original_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.preview_card = ImageCard(stage, "Preview", "Adjust a control to preview")
        self.preview_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.chart = ChartPanel(stage, "Luma histogram of the preview")
        self.chart.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.chart.show_message("The histogram appears with the preview.")

        side = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_color=COLORS["border"],
            border_width=1,
        )
        side.grid(row=0, column=1, sticky="nsew")
        params = app.session.params
        SectionLabel(side, "Noise reduction").pack(anchor="w", padx=8, pady=(8, 4))
        Hint(
            side,
            "Median first, then Gaussian. The median removes impulses without smearing them; the Gaussian attenuates the remaining high frequencies.",
        ).pack(anchor="w", padx=8, pady=(0, 6))
        self.median = SliderRow(
            side, "Median window", 1, 9, int(params["median_size"]), 8,
            lambda value: f"{int(round(value))} px",
            self._on_median,
            "Odd width of the median neighborhood. 1 turns the filter off.",
        )
        self.median.pack(fill="x", padx=8)
        self.sigma = SliderRow(
            side, "Gaussian σ", 0.0, 5.0, float(params["gaussian_sigma"]), 50,
            lambda value: f"{value:.1f} px",
            self._on_sigma,
            "Standard deviation of the Gaussian kernel, in pixels. 0 turns it off. A value near 0.8 is a mild denoise.",
        )
        self.sigma.pack(fill="x", padx=8)

        SectionLabel(side, "Contrast").pack(anchor="w", padx=8, pady=(10, 4))
        self.mode = ctk.CTkOptionMenu(
            side,
            values=["none", "stretch", "gamma", "equalize"],
            command=self._on_mode,
            fg_color=COLORS["stage"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            dropdown_fg_color=COLORS["card"],
            text_color=COLORS["text"],
        )
        self.mode.set(str(params["contrast_mode"]))
        self.mode.pack(fill="x", padx=8, pady=4)
        Hint(
            side,
            "Stretch maps percentiles to the full range. Gamma is a power law. Equalize flattens the luma histogram and rescales RGB so hue stays put.",
        ).pack(anchor="w", padx=8, pady=(0, 6))
        self.low = SliderRow(
            side, "Stretch low percentile", 0.0, 20.0, float(params["stretch_low"]), 40,
            lambda value: f"{value:.1f}",
            lambda value: self._set("stretch_low", round(value, 2)),
            "Lower anchor of the linear stretch. Used when the contrast mode is stretch.",
        )
        self.low.pack(fill="x", padx=8)
        self.high = SliderRow(
            side, "Stretch high percentile", 80.0, 100.0, float(params["stretch_high"]), 40,
            lambda value: f"{value:.1f}",
            lambda value: self._set("stretch_high", round(value, 2)),
            "Upper anchor of the linear stretch. It must stay above the low percentile.",
        )
        self.high.pack(fill="x", padx=8)
        self.gamma = SliderRow(
            side, "Gamma", 0.3, 3.0, float(params["gamma"]), 54,
            lambda value: f"{value:.2f}",
            lambda value: self._set("gamma", round(value, 2)),
            "Exponent of C' = C^γ on intensities in [0, 1]. Below 1 lifts shadows. Used when the contrast mode is gamma.",
        )
        self.gamma.pack(fill="x", padx=8)

        self.dirty = ctk.CTkLabel(side, text="", font=FONT_SMALL, text_color=COLORS["warning"], wraplength=320, justify="left")
        self.dirty.pack(anchor="w", padx=8, pady=8)
        primary_button(side, "Apply to workspace", self._apply).pack(fill="x", padx=8, pady=4)
        secondary_button(side, "Reset preprocessing", self._reset).pack(fill="x", padx=8, pady=(4, 12))

    def sync_from_state(self) -> None:
        params = self.app.session.params
        self.median.set_value(float(params["median_size"]))
        self.sigma.set_value(float(params["gaussian_sigma"]))
        self.mode.set(str(params["contrast_mode"]))
        self.low.set_value(float(params["stretch_low"]))
        self.high.set_value(float(params["stretch_high"]))
        self.gamma.set_value(float(params["gamma"]))

    def refresh_views(self) -> None:
        state = self.app.session
        self.original_card.show(state.original)
        shown = self._preview if self._preview is not None else state.working
        self.preview_card.show(shown)
        self._update_dirty()

    def on_image_replaced(self) -> None:
        self._generation += 1
        self._preview = None if self.app.session.original is None else self.app.session.original
        self._preview_params = dict(self.app.session.applied_preprocess)
        self.preview_card.show(self._preview)
        if self._preview is not None:
            centers, counts = luminance_histogram(self._preview)
            self.chart.histogram(centers, counts, "Luma histogram of the working image")
        else:
            self.chart.show_message("The histogram appears with the preview.")
        self._update_dirty()

    def on_workspace_reset(self) -> None:
        self.on_image_replaced()

    def _snapshot(self) -> dict:
        return {key: self.app.session.params[key] for key in PREPROCESS_KEYS}

    def _set(self, key: str, value) -> None:
        self.app.session.params[key] = value
        self._schedule()

    def _on_median(self, value: float) -> None:
        snapped = int(round(value))
        if snapped % 2 == 0:
            snapped -= 1
        snapped = max(1, snapped)
        if abs(snapped - value) > 0.4:
            self.median.set_value(snapped)
        self._set("median_size", snapped)

    def _on_sigma(self, value: float) -> None:
        self._set("gaussian_sigma", round(value, 2))

    def _on_mode(self, value: str) -> None:
        self._set("contrast_mode", value)

    def _schedule(self) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(280, self._start_preview)
        self._update_dirty()

    def _start_preview(self) -> None:
        self._job = None
        original = self.app.session.original
        if original is None:
            return
        if self.app.busy:
            self._schedule()
            return
        params = self._snapshot()
        self._generation += 1
        generation = self._generation

        def work():
            image = preprocess(original, params)
            centers, counts = luminance_histogram(image)
            return image, centers, counts, params

        def done(result):
            if generation != self._generation:
                return
            image, centers, counts, used = result
            self._preview = image
            self._preview_params = used
            self.preview_card.show(image)
            self.chart.histogram(centers, counts, "Luma histogram of the preview")
            self._update_dirty()

        self.app.spawn(work, done)

    def _update_dirty(self) -> None:
        if not self.app.session.has_image():
            self.dirty.configure(text="")
            return
        if self.app.session.preprocess_is_dirty():
            self.dirty.configure(text="This preview is not the working image yet. Apply it before Fourier analysis or segmentation.")
        else:
            self.dirty.configure(text="The working image matches these controls.", text_color=COLORS["success"])
            return
        self.dirty.configure(text_color=COLORS["warning"])

    def _apply(self) -> None:
        if not self.app.require_image():
            return
        params = self._snapshot()
        if self._preview is not None and self._preview_params == params:
            self.app.commit_preprocessing(self._preview, params)
            self._update_dirty()
            return
        original = self.app.session.original

        def work():
            return preprocess(original, params)

        def done(image):
            self._preview = image
            self._preview_params = params
            self.preview_card.show(image)
            centers, counts = luminance_histogram(image)
            self.chart.histogram(centers, counts, "Luma histogram of the preview")
            self.app.commit_preprocessing(image, params)
            self._update_dirty()

        self.app.run_task("Applying preprocessing", work, done)

    def _reset(self) -> None:
        if self.app.session.original is None:
            for key in PREPROCESS_KEYS:
                self.app.session.params[key] = DEFAULT_PARAMS[key]
            self.sync_from_state()
            return
        for key in PREPROCESS_KEYS:
            self.app.session.params[key] = DEFAULT_PARAMS[key]
        self.sync_from_state()
        params = self._snapshot()
        image = self.app.session.original.copy()
        self._preview = image
        self._preview_params = params
        self._generation += 1
        self.preview_card.show(image)
        centers, counts = luminance_histogram(image)
        self.chart.histogram(centers, counts, "Luma histogram of the preview")
        self.app.commit_preprocessing(image, params)
        self._update_dirty()
