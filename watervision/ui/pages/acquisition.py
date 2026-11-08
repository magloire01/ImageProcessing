"""Load, inspect, zoom, and reset.

Zoom is a single scale factor. At 100% the photograph is fitted to the
canvas; the wheel and the slider increase that factor and the canvas
scrolls. Dragging pans the zoomed image.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.io_utils import image_info
from watervision.theme import COLORS, FONT_BODY, FONT_CARD, FONT_SMALL
from watervision.ui.tooltip import Tooltip
from watervision.ui.widgets import SliderRow, ZoomView, danger_button, primary_button, secondary_button


def _bytes(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


class AcquisitionPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        self.viewer = ZoomView(self)
        self.viewer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.viewer.on_zoom_changed = self._on_view_zoom

        side = ctk.CTkFrame(
            self,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        side.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(side, text="Source file", font=FONT_CARD, text_color=COLORS["text"]).pack(
            anchor="w", padx=16, pady=(16, 8)
        )
        self.info = ctk.CTkLabel(
            side,
            text="No image loaded.",
            font=FONT_BODY,
            text_color=COLORS["muted"],
            justify="left",
            anchor="w",
            wraplength=280,
        )
        self.info.pack(anchor="w", padx=16, pady=(0, 12))

        primary_button(side, "Load image", app.open_image).pack(fill="x", padx=16, pady=4)
        secondary_button(side, "Demonstration scene", app.load_demo).pack(fill="x", padx=16, pady=4)
        fit = secondary_button(side, "Fit to view", self._fit)
        fit.pack(fill="x", padx=16, pady=4)
        Tooltip(fit, "Set the zoom so the whole image is visible. 100% on the slider means fit, not one image pixel per screen pixel.")

        self.zoom = SliderRow(
            side,
            "Zoom",
            1.0,
            8.0,
            1.0,
            70,
            lambda value: f"{value * 100:.0f}%",
            self._on_slider_zoom,
            "Drag, or use the mouse wheel over the image. Drag the picture itself to pan when it is larger than the view.",
        )
        self.zoom.pack(fill="x", padx=16, pady=(16, 4))

        note = ctk.CTkLabel(
            side,
            text="Accepted files: JPG, JPEG, PNG, TIFF, BMP. A multi-page TIFF contributes its first frame.",
            font=FONT_SMALL,
            text_color=COLORS["muted"],
            wraplength=280,
            justify="left",
        )
        note.pack(anchor="w", padx=16, pady=12)
        danger_button(side, "Reset workspace", app.reset_workspace).pack(fill="x", padx=16, pady=(8, 16), side="bottom")

    def sync_from_state(self) -> None:
        return

    def refresh_views(self) -> None:
        state = self.app.session
        # Avoid resetting the user's zoom on every status refresh. Only
        # replace the raster when the acquisition identity changes.
        token = (id(state.original) if state.original is not None else 0, state.filename)
        if token != getattr(self, "_shown_token", None):
            self._shown_token = token
            self.viewer.show(state.original)
            self.zoom.set_value(1.0)
        if state.original is None:
            self.info.configure(text="No image loaded.")
            return
        info = image_info(state.original, state.path)
        lines = [
            state.filename or "Untitled",
            f"{info['width']} × {info['height']} px",
            f"{info['dtype']}, {info['channels']} channels",
            f"{info['megapixels']:.2f} megapixels",
            f"File size {_bytes(info['bytes'])}",
        ]
        if state.load_note:
            lines.append(state.load_note)
        if state.preprocess_is_dirty():
            lines.append("Preprocessing controls differ from the working image.")
        elif state.applied_preprocess.get("contrast_mode", "none") != "none" or float(state.applied_preprocess.get("gaussian_sigma", 0)) > 0:
            lines.append("The working image has been preprocessed.")
        self.info.configure(text="\n".join(lines))

    def _fit(self) -> None:
        self.viewer.set_multiplier(1.0)
        self.zoom.set_value(1.0)

    def _on_slider_zoom(self, value: float) -> None:
        self.viewer.set_multiplier(value)

    def _on_view_zoom(self, value: float) -> None:
        self.zoom.set_value(value)

    def on_workspace_reset(self) -> None:
        self._shown_token = None
