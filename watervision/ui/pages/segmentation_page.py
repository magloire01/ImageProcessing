"""Three segmentation methods and the morphological cleanup.

HSV is a hard color test. The Fourier mask is a smooth-region candidate and
is labeled as such. The hybrid score is the method intended for the area
estimate. Morphology is shared so the comparison is about the decision, not
about leftover speckle.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.io_utils import mask_to_rgb
from watervision.processing.metrics import METHOD_LABELS
from watervision.theme import COLORS, FONT_SMALL
from watervision.ui.widgets import Hint, ImageCard, SectionLabel, SliderRow, SwitchRow, primary_button, secondary_button

_METHOD_TO_LABEL = {"hsv": "HSV", "fourier": "Fourier", "hybrid": "Hybrid"}
_LABEL_TO_METHOD = {label: key for key, label in _METHOD_TO_LABEL.items()}


class SegmentationPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        stage = ctk.CTkFrame(self, fg_color="transparent")
        stage.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        for index in (0, 1):
            stage.grid_columnconfigure(index, weight=1)
            stage.grid_rowconfigure(index, weight=1)
        self.original_card = ImageCard(stage, "Working image")
        self.original_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.hsv_card = ImageCard(stage, "HSV mask", "Color threshold")
        self.hsv_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.fourier_card = ImageCard(stage, "Fourier candidate", "Smooth regions, not proof of water")
        self.fourier_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        self.hybrid_card = ImageCard(stage, "Hybrid mask", "Color and texture")
        self.hybrid_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))

        side = ctk.CTkFrame(
            self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_rowconfigure(0, weight=1)
        side.grid_columnconfigure(0, weight=1)
        params = app.session.params

        self.tabs = ctk.CTkTabview(
            side,
            fg_color=COLORS["card"],
            segmented_button_fg_color=COLORS["stage"],
            segmented_button_selected_color=COLORS["blue"],
            segmented_button_selected_hover_color=COLORS["blue_hover"],
            segmented_button_unselected_color=COLORS["stage"],
            segmented_button_unselected_hover_color="#1B3048",
            text_color=COLORS["text"],
        )
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 0))
        self._build_hsv(self.tabs.add("HSV"), params)
        self._build_frequency(self.tabs.add("Frequency"), params)
        self._build_hybrid(self.tabs.add("Hybrid"), params)
        self._build_morphology(self.tabs.add("Morphology"), params)

        bar = ctk.CTkFrame(side, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        self.active = ctk.CTkSegmentedButton(
            bar,
            values=["HSV", "Fourier", "Hybrid"],
            command=self._on_active,
            selected_color=COLORS["blue"],
            selected_hover_color=COLORS["blue_hover"],
            unselected_color=COLORS["stage"],
            text_color=COLORS["text"],
        )
        self.active.set(_METHOD_TO_LABEL[str(params["active_method"])])
        self.active.pack(fill="x", pady=(0, 6))
        Hint(bar, "The highlighted method is the one used for area, the dashboard overlay, and the default export.").pack(
            anchor="w", pady=(0, 6)
        )
        primary_button(bar, "Run segmentation", app.run_segmentation).pack(fill="x", pady=3)
        secondary_button(bar, "Reset parameters", app.reset_segmentation_params).pack(fill="x", pady=3)
        self.time_label = ctk.CTkLabel(bar, text="", font=FONT_SMALL, text_color=COLORS["muted"])
        self.time_label.pack(anchor="w", pady=(4, 0))

    def _scroll(self, tab) -> ctk.CTkScrollableFrame:
        frame = ctk.CTkScrollableFrame(tab, fg_color=COLORS["card"])
        frame.pack(fill="both", expand=True)
        return frame

    def _build_hsv(self, tab, params) -> None:
        frame = self._scroll(tab)
        Hint(
            frame,
            "Default detection keeps blue pixels only: hue from 185° to 255°. Green and yellow land fall outside that arc. The second band and the dark-water rule are off until you turn them on.",
        ).pack(anchor="w", pady=(0, 6))
        self.h_min = self._slider(frame, "Hue minimum", 0, 360, params["h_min"], 360, "{:.0f}°", "h_min")
        self.h_max = self._slider(frame, "Hue maximum", 0, 360, params["h_max"], 360, "{:.0f}°", "h_max")
        self.s_min = self._unit(frame, "Saturation minimum", params["s_min"], "s_min")
        self.s_max = self._unit(frame, "Saturation maximum", params["s_max"], "s_max")
        self.v_min = self._unit(frame, "Value minimum", params["v_min"], "v_min")
        self.v_max = self._unit(frame, "Value maximum", params["v_max"], "v_max")
        self.secondary = SwitchRow(
            frame, "Second hue band (turbid water)", bool(params["use_secondary_hue"]),
            lambda value: self._set("use_secondary_hue", value),
            "Enable a separate hue interval for brown or sediment-laden water.",
        )
        self.secondary.pack(anchor="w", pady=4)
        self.h2_min = self._slider(frame, "Second hue minimum", 0, 360, params["h2_min"], 360, "{:.0f}°", "h2_min")
        self.h2_max = self._slider(frame, "Second hue maximum", 0, 360, params["h2_max"], 360, "{:.0f}°", "h2_max")
        self.s2_min = self._unit(frame, "Second saturation min", params["s2_min"], "s2_min")
        self.s2_max = self._unit(frame, "Second saturation max", params["s2_max"], "s2_max")
        self.v2_min = self._unit(frame, "Second value min", params["v2_min"], "v2_min")
        self.v2_max = self._unit(frame, "Second value max", params["v2_max"], "v2_max")
        self.dark = SwitchRow(
            frame, "Include dark low-saturation water", bool(params["include_dark_water"]),
            lambda value: self._set("include_dark_water", value),
            "Grey water has no stable hue. Accept it only when it is dark and weakly saturated.",
        )
        self.dark.pack(anchor="w", pady=4)
        self.dark_v = self._unit(frame, "Dark value maximum", params["dark_v_max"], "dark_v_max")
        self.dark_s = self._unit(frame, "Dark saturation maximum", params["dark_s_max"], "dark_s_max")

    def _build_frequency(self, tab, params) -> None:
        frame = self._scroll(tab)
        Hint(
            frame,
            "A pixel is a candidate when its local high-frequency energy is below the threshold. Roads, sand, and shadow pass this test too. Do not read this mask as detected water.",
        ).pack(anchor="w", pady=(0, 6))
        self.texture = self._slider(
            frame, "Texture threshold", 0.05, 0.9, params["texture_threshold"], 85, "{:.2f}", "texture_threshold",
            "Hard cutoff for the Fourier mask, and the center of the logistic used by the hybrid score.",
        )
        self.steepness = self._slider(
            frame, "Logistic steepness", 4, 30, params["texture_steepness"], 26, "{:.0f}", "texture_steepness",
            "How quickly the hybrid texture score falls once energy passes the threshold.",
        )
        self.patch_window = self._slider(
            frame, "Patch size", 8, 64, params["patch_window"], 56, "{:.0f} px", "patch_window"
        )
        self.patch_cutoff = self._slider(
            frame, "High-frequency split", 0.1, 0.8, params["patch_cutoff"], 70, "{:.2f}", "patch_cutoff"
        )

    def _build_hybrid(self, tab, params) -> None:
        frame = self._scroll(tab)
        Hint(
            frame,
            "Score = color weight × color + texture weight × smoothness + homogeneity weight × homogeneity. The weights are rescaled so they sum to 1. With the defaults, smoothness alone stays below the decision threshold.",
        ).pack(anchor="w", pady=(0, 6))
        self.w_color = self._slider(frame, "Color weight", 0, 1, params["weight_color"], 100, "{:.2f}", "weight_color")
        self.w_texture = self._slider(frame, "Texture weight", 0, 1, params["weight_texture"], 100, "{:.2f}", "weight_texture")
        self.w_hom = self._slider(
            frame, "Homogeneity weight", 0, 1, params["weight_homogeneity"], 100, "{:.2f}", "weight_homogeneity"
        )
        self.use_hom = SwitchRow(
            frame, "Use local homogeneity", bool(params["use_homogeneity"]),
            lambda value: self._set("use_homogeneity", value),
            "Homogeneity is 1 when the local standard deviation is far below the reference. It describes spread, not which frequencies carry it.",
        )
        self.use_hom.pack(anchor="w", pady=4)
        self.decision = self._slider(
            frame, "Decision threshold", 0.05, 0.95, params["decision_threshold"], 90, "{:.2f}", "decision_threshold",
            "Pixels whose hybrid score is at least this value are labeled water.",
        )
        self.hue_soft = self._slider(
            frame, "Hue softness", 4, 40, params["hue_softness"], 36, "{:.0f}°", "hue_softness",
            "Gaussian width of the hue membership outside the selected interval.",
        )
        self.std_ref = self._slider(
            frame, "Homogeneity reference σ", 0.02, 0.25, params["homogeneity_std_ref"], 46, "{:.2f}",
            "homogeneity_std_ref",
            "Local standard deviation, on the 0–1 luma scale, that maps to homogeneity 0.",
        )
        self.weight_label = ctk.CTkLabel(frame, text="", font=FONT_SMALL, text_color=COLORS["accent"], wraplength=320, justify="left")
        self.weight_label.pack(anchor="w", pady=6)

    def _build_morphology(self, tab, params) -> None:
        frame = self._scroll(tab)
        Hint(
            frame,
            "Order is fixed: opening, then closing, then hole filling, then removal of small components. Opening deletes thin false positives before closing can join them into a lake.",
        ).pack(anchor="w", pady=(0, 6))
        self.radius = self._slider(
            frame, "Structuring-element radius", 0, 8, params["morph_radius"], 8, "{:.0f} px", "morph_radius",
            "Disk radius. 0 makes opening and closing the identity.",
        )
        self.do_open = SwitchRow(frame, "Opening", bool(params["do_opening"]), lambda value: self._set("do_opening", value))
        self.do_close = SwitchRow(frame, "Closing", bool(params["do_closing"]), lambda value: self._set("do_closing", value))
        self.do_holes = SwitchRow(
            frame, "Fill enclosed holes", bool(params["do_fill_holes"]), lambda value: self._set("do_fill_holes", value),
            "Fills background pockets that cannot reach the image border. A bay that touches the border is left open.",
        )
        self.do_small = SwitchRow(
            frame, "Remove small regions", bool(params["do_remove_small"]), lambda value: self._set("do_remove_small", value)
        )
        for switch in (self.do_open, self.do_close, self.do_holes, self.do_small):
            switch.pack(anchor="w")
        self.min_region = self._slider(
            frame, "Minimum region size", 0, 2000, params["min_region_pixels"], 200, "{:.0f} px", "min_region_pixels",
            "Connected components smaller than this many pixels are deleted. This is a pixel count, not a ground area.",
        )

    def _slider(self, parent, label, start, end, initial, steps, fmt, key, tooltip=""):
        row = SliderRow(
            parent, label, start, end, float(initial), steps,
            lambda value, pattern=fmt: pattern.format(value),
            lambda value, name=key: self._set_numeric(name, value),
            tooltip,
        )
        row.pack(fill="x", pady=2)
        return row

    def _unit(self, parent, label, initial, key):
        return self._slider(parent, label, 0, 1, initial, 100, "{:.2f}", key)

    def _set(self, key: str, value) -> None:
        self.app.session.params[key] = value

    def _set_numeric(self, key: str, value: float) -> None:
        if key in ("patch_window", "morph_radius", "min_region_pixels", "texture_steepness", "hue_softness"):
            stored = int(round(value))
        else:
            stored = round(float(value), 3)
        self.app.session.params[key] = stored

    def _on_active(self, label: str) -> None:
        self.app.session.params["active_method"] = _LABEL_TO_METHOD[label]
        self.app.refresh_current()

    def sync_from_state(self) -> None:
        params = self.app.session.params
        mapping = {
            self.h_min: "h_min", self.h_max: "h_max",
            self.s_min: "s_min", self.s_max: "s_max",
            self.v_min: "v_min", self.v_max: "v_max",
            self.h2_min: "h2_min", self.h2_max: "h2_max",
            self.s2_min: "s2_min", self.s2_max: "s2_max",
            self.v2_min: "v2_min", self.v2_max: "v2_max",
            self.dark_v: "dark_v_max", self.dark_s: "dark_s_max",
            self.texture: "texture_threshold", self.steepness: "texture_steepness",
            self.patch_window: "patch_window", self.patch_cutoff: "patch_cutoff",
            self.w_color: "weight_color", self.w_texture: "weight_texture", self.w_hom: "weight_homogeneity",
            self.decision: "decision_threshold", self.hue_soft: "hue_softness", self.std_ref: "homogeneity_std_ref",
            self.radius: "morph_radius", self.min_region: "min_region_pixels",
        }
        for widget, key in mapping.items():
            widget.set_value(float(params[key]))
        self.secondary.set_value(bool(params["use_secondary_hue"]))
        self.dark.set_value(bool(params["include_dark_water"]))
        self.use_hom.set_value(bool(params["use_homogeneity"]))
        self.do_open.set_value(bool(params["do_opening"]))
        self.do_close.set_value(bool(params["do_closing"]))
        self.do_holes.set_value(bool(params["do_fill_holes"]))
        self.do_small.set_value(bool(params["do_remove_small"]))
        self.active.set(_METHOD_TO_LABEL[str(params["active_method"])])

    def refresh_views(self) -> None:
        state = self.app.session
        self.original_card.show(state.working)
        self._show_mask(self.hsv_card, "hsv")
        self._show_mask(self.fourier_card, "fourier")
        self._show_mask(self.hybrid_card, "hybrid")
        if state.weights_used:
            weights = state.weights_used
            self.weight_label.configure(
                text=(
                    f"Last run: color {weights['color']:.2f}, "
                    f"texture {weights['texture']:.2f}, "
                    f"homogeneity {weights['homogeneity']:.2f}"
                )
            )
        else:
            self.weight_label.configure(text="Weights are normalized to 1 when segmentation runs.")
        elapsed = state.timings.get("segmentation")
        self.time_label.configure(
            text="" if elapsed is None else f"Last segmentation {elapsed:.2f} s. Masks include the current morphology."
        )

    def _show_mask(self, card: ImageCard, method: str) -> None:
        mask = self.app.session.mask_clean.get(method)
        metrics = self.app.session.metrics.get(method)
        if mask is None:
            card.show(None)
            card.set_subtitle("")
            return
        card.show(mask_to_rgb(mask))
        if metrics:
            card.set_subtitle(f"{metrics['percentage']:.1f}%  ·  {metrics['regions']} regions")
        else:
            card.set_subtitle(METHOD_LABELS[method])
