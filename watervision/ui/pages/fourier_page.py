"""Full-image Fourier transform and the local frequency map.

The centered spectrum and the filtered reconstruction are shown beside the
photograph that was transformed (the working image). The local energy map
is a texture measurement. Bright means high-frequency energy. It is not a
water mask; the segmentation page is where that distinction is enforced.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.fourier import energy_to_rgb
from watervision.theme import COLORS
from watervision.ui.widgets import Hint, ImageCard, SectionLabel, SliderRow, primary_button


class FourierPage(ctk.CTkFrame):
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
        self.spectrum_card = ImageCard(stage, "Log magnitude spectrum", "Compute the transform")
        self.spectrum_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.filtered_card = ImageCard(stage, "Filtered image", "Choose a cutoff")
        self.filtered_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        self.energy_card = ImageCard(stage, "Local high-frequency energy", "Not computed")
        self.energy_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))

        side = ctk.CTkScrollableFrame(
            self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        side.grid(row=0, column=1, sticky="nsew")
        params = app.session.params

        SectionLabel(side, "Frequency filter").pack(anchor="w", padx=8, pady=(8, 4))
        Hint(
            side,
            "Low-pass keeps a disk around the center (the mean and the slow shading). High-pass removes that disk. The center of the spectrum is zero frequency.",
        ).pack(anchor="w", padx=8)
        self.kind = ctk.CTkSegmentedButton(
            side,
            values=["low-pass", "high-pass"],
            command=self._on_kind,
            selected_color=COLORS["blue"],
            selected_hover_color=COLORS["blue_hover"],
            unselected_color=COLORS["stage"],
            unselected_hover_color="#1B3048",
            text_color=COLORS["text"],
        )
        self.kind.set(str(params["filter_kind"]))
        self.kind.pack(fill="x", padx=8, pady=6)
        self.response = ctk.CTkSegmentedButton(
            side,
            values=["butterworth", "ideal"],
            command=self._on_response,
            selected_color=COLORS["blue"],
            selected_hover_color=COLORS["blue_hover"],
            unselected_color=COLORS["stage"],
            unselected_hover_color="#1B3048",
            text_color=COLORS["text"],
        )
        self.response.set(str(params["filter_response"]))
        self.response.pack(fill="x", padx=8, pady=4)
        self.cutoff = SliderRow(
            side, "Cutoff fraction", 0.02, 0.95, float(params["cutoff"]), 93,
            lambda value: f"{value:.2f}",
            lambda value: self._set_filter("cutoff", round(value, 3)),
            "Fraction of the distance from the spectrum center to its corner. 1 would include every frequency.",
        )
        self.cutoff.pack(fill="x", padx=8)
        self.order = SliderRow(
            side, "Butterworth order", 1, 8, int(params["butterworth_order"]), 7,
            lambda value: f"{int(round(value))}",
            self._on_order,
            "Order n in (D/D0)^(2n). Higher n is closer to the ideal disk and rings more in the image.",
        )
        self.order.pack(fill="x", padx=8)
        self.cmap = ctk.CTkOptionMenu(
            side,
            values=["magma", "viridis", "gray", "inferno"],
            command=self._on_cmap,
            fg_color=COLORS["stage"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            dropdown_fg_color=COLORS["card"],
            text_color=COLORS["text"],
        )
        self.cmap.set(str(params["spectrum_cmap"]))
        self.cmap.pack(fill="x", padx=8, pady=6)

        SectionLabel(side, "Local frequency").pack(anchor="w", padx=8, pady=(12, 4))
        Hint(
            side,
            "Each patch is windowed, its mean is removed, and the share of spectral energy outside a low-frequency disk is recorded. Smooth sand scores like smooth water. This map alone is not evidence of water.",
        ).pack(anchor="w", padx=8, pady=(0, 6))
        self.window = SliderRow(
            side, "Patch size", 8, 64, int(params["patch_window"]), 56,
            lambda value: f"{int(round(value))} px",
            self._on_window,
            "Side of the square patch, in pixels. Larger patches describe coarser texture.",
        )
        self.window.pack(fill="x", padx=8)
        self.patch_cutoff = SliderRow(
            side, "High-frequency split", 0.1, 0.8, float(params["patch_cutoff"]), 70,
            lambda value: f"{value:.2f}",
            lambda value: self._set_patch("patch_cutoff", round(value, 3)),
            "Fraction of the patch Nyquist radius. Energy beyond it counts as high frequency.",
        )
        self.patch_cutoff.pack(fill="x", padx=8)

        primary_button(side, "Compute spectrum", app.compute_spectrum).pack(fill="x", padx=8, pady=(12, 4))
        primary_button(side, "Compute local frequency map", app.compute_local_energy).pack(fill="x", padx=8, pady=(4, 14))

    def sync_from_state(self) -> None:
        params = self.app.session.params
        self.kind.set(str(params["filter_kind"]))
        self.response.set(str(params["filter_response"]))
        self.cutoff.set_value(float(params["cutoff"]))
        self.order.set_value(float(params["butterworth_order"]))
        self.cmap.set(str(params["spectrum_cmap"]))
        self.window.set_value(float(params["patch_window"]))
        self.patch_cutoff.set_value(float(params["patch_cutoff"]))

    def refresh_views(self) -> None:
        state = self.app.session
        self.original_card.show(state.working)
        self.spectrum_card.show(state.spectrum_rgb)
        self.spectrum_card.set_subtitle("log(1 + |F|), display stretch only" if state.spectrum_rgb is not None else "")
        self.filtered_card.show(state.filtered_rgb)
        self.filtered_card.set_subtitle(state.filter_label)
        if state.energy is not None and state.energy_key == self.app.energy_key():
            self.energy_card.show(energy_to_rgb(state.energy, str(state.params["spectrum_cmap"])))
            self.energy_card.set_subtitle("Bright = textured")
        else:
            self.energy_card.show(None)
            self.energy_card.set_subtitle("Settings changed" if state.energy is not None else "Not computed")

    def on_enter(self) -> None:
        state = self.app.session
        if self.app.busy or not state.has_image():
            return
        if state.spectrum_revision != state.revision:
            self.app.compute_spectrum()

    def _set_filter(self, key: str, value) -> None:
        self.app.session.params[key] = value
        self.app.schedule_filter()

    def _on_kind(self, value: str) -> None:
        self._set_filter("filter_kind", value)

    def _on_response(self, value: str) -> None:
        self._set_filter("filter_response", value)

    def _on_order(self, value: float) -> None:
        self._set_filter("butterworth_order", int(round(value)))

    def _on_cmap(self, value: str) -> None:
        self._set_filter("spectrum_cmap", value)

    def _set_patch(self, key: str, value) -> None:
        self.app.session.params[key] = value
        # The cached map no longer matches. The view update is local and cheap.
        if self.app.current == "fourier":
            self.refresh_views()

    def _on_window(self, value: float) -> None:
        self._set_patch("patch_window", int(round(value)))
