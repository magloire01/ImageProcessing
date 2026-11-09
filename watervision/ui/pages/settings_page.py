"""Workspace preferences that are not tied to one processing button.

The colormap is the same control as on the Fourier page. The window cap
limits how many local spectra are computed, which is what keeps that step
responsive on a large photograph. Restoring defaults puts every parameter
back and returns the working image to the original file.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision import __version__
from watervision.theme import COLORS, FONT_BODY, FONT_CARD, FONT_SMALL
from watervision.ui.widgets import Hint, SectionLabel, SliderRow, danger_button

_ABOUT = """WaterVision estimates the water surface in one photograph.

The 2D discrete Fourier transform describes the whole image as a sum of spatial frequencies. After fftshift, the center is the mean (zero frequency) and the corners are the finest oscillation the pixel grid can hold. A low-pass multiplies that spectrum by a disk or a Butterworth function and transforms back, which blurs. A high-pass removes the mean and the slow shading, which leaves edges and texture.

Local high-frequency energy repeats a small version of that measurement inside a sliding patch. The ratio is the share of patch energy outside a low-frequency disk, after a Hann window and after the patch mean is removed. Calm water is often smooth. So are sand, pavement, and shadow. The frequency mask is therefore a candidate, not a detection.

The hybrid score adds a soft HSV membership so color has to agree. By default only blue pixels are kept (hue 185° to 255°). A second ochre band and a dark-water rule exist, and both stay off until you enable them.

Morphology then cleans the binary sets. Opening removes thin false positives before closing is allowed to bridge gaps. Holes are filled only when they do not touch the image border, then any filled pixel that is not blue is removed, so a continent surrounded by ocean stays out of the mask. Small components are removed by pixel count.

Area is a pixel count and a percentage unless a pixel width and height in meters are entered. Without that scale the result is not a ground area.

No OpenCV and no trained model are used. The arithmetic is NumPy and SciPy.
"""


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(
            self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        SectionLabel(side, "Local spectra").pack(anchor="w", padx=16, pady=(16, 4))
        Hint(
            side,
            "Maximum number of patches along each axis. The stride grows so a large image does not request one transform per pixel.",
        ).pack(anchor="w", padx=16)
        self.windows = SliderRow(
            side,
            "Frequency windows per axis",
            8,
            80,
            int(app.session.params["max_frequency_windows"]),
            72,
            lambda value: f"{int(round(value))}",
            lambda value: self._set_windows(value),
        )
        self.windows.pack(fill="x", padx=16, pady=8)
        SectionLabel(side, "Spectrum colors").pack(anchor="w", padx=16, pady=(8, 4))
        self.cmap = ctk.CTkOptionMenu(
            side,
            values=["magma", "viridis", "gray", "inferno"],
            command=self._set_cmap,
            fg_color=COLORS["stage"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            dropdown_fg_color=COLORS["card"],
            text_color=COLORS["text"],
        )
        self.cmap.set(str(app.session.params["spectrum_cmap"]))
        self.cmap.pack(fill="x", padx=16, pady=8)
        danger_button(side, "Restore default parameters", app.restore_defaults).pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(
            side,
            text=f"WaterVision {__version__}",
            font=FONT_SMALL,
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=16, pady=(4, 16))

        about = ctk.CTkFrame(
            self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        about.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(about, text="What the measurements mean", font=FONT_CARD, text_color=COLORS["text"]).pack(
            anchor="w", padx=16, pady=(14, 6)
        )
        box = ctk.CTkTextbox(
            about,
            fg_color=COLORS["stage"],
            text_color=COLORS["text"],
            font=FONT_BODY,
            wrap="word",
            activate_scrollbars=True,
        )
        box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        box.insert("1.0", _ABOUT.strip())
        box.configure(state="disabled")

    def _set_windows(self, value: float) -> None:
        self.app.session.params["max_frequency_windows"] = int(round(value))

    def _set_cmap(self, value: str) -> None:
        self.app.session.params["spectrum_cmap"] = value
        self.app.schedule_filter()

    def sync_from_state(self) -> None:
        params = self.app.session.params
        self.windows.set_value(float(params["max_frequency_windows"]))
        self.cmap.set(str(params["spectrum_cmap"]))

    def refresh_views(self) -> None:
        return
