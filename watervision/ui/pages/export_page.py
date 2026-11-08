"""Exports. Each button writes a real file, or explains which step is missing.

The mask selector defaults to the active segmentation method. The PDF and
the CSV can be written from an image alone; sections that were not computed
are labeled as such instead of being invented.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.theme import COLORS, FONT_BODY, FONT_CARD, FONT_SMALL
from watervision.ui.widgets import primary_button

_LABELS = ("HSV", "Fourier", "Hybrid")
_KEYS = {"HSV": "hsv", "Fourier": "fourier", "Hybrid": "hybrid"}


class ExportPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        intro = ctk.CTkFrame(
            self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        intro.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(intro, text="Mask used for image exports", font=FONT_CARD, text_color=COLORS["text"]).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        ctk.CTkLabel(
            intro,
            text="Binary masks are 0 and 255. The highlight is drawn on the working image, the one that was segmented.",
            font=FONT_SMALL,
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=16)
        self.method = ctk.CTkSegmentedButton(
            intro,
            values=list(_LABELS),
            selected_color=COLORS["blue"],
            selected_hover_color=COLORS["blue_hover"],
            unselected_color=COLORS["stage"],
            text_color=COLORS["text"],
        )
        self.method.set("Hybrid")
        self.method.pack(anchor="w", padx=16, pady=(8, 14))

        actions = (
            ("Save binary mask", "PNG label image of the selected method after morphology.", app.export_mask),
            ("Save highlighted image", "Working image with water filled in cyan and the boundary in white.", app.export_overlay),
            ("Save Fourier spectrum", "The log-magnitude picture. The transform is computed first if it is missing.", app.export_spectrum),
            ("Export CSV", "Coverage, scores, and every parameter. Uncomputed sections are left blank.", app.export_csv),
            ("Export PDF report", "Image, masks, area, spectrum, scores, and the method statement.", app.export_pdf),
        )
        for index, (title, body, command) in enumerate(actions):
            card = ctk.CTkFrame(
                self, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
            )
            row, column = divmod(index, 2)
            card.grid(row=row + 1, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0), pady=6)
            self.grid_rowconfigure(row + 1, weight=1)
            ctk.CTkLabel(card, text=title, font=FONT_CARD, text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 4))
            ctk.CTkLabel(
                card, text=body, font=FONT_BODY, text_color=COLORS["muted"], wraplength=420, justify="left"
            ).pack(anchor="w", padx=16)
            primary_button(card, title, command).pack(anchor="w", padx=16, pady=(10, 14))

    def selected_method(self) -> str:
        return _KEYS[self.method.get()]

    def sync_from_state(self) -> None:
        labels = {"hsv": "HSV", "fourier": "Fourier", "hybrid": "Hybrid"}
        self.method.set(labels.get(str(self.app.session.params.get("active_method", "hybrid")), "Hybrid"))

    def refresh_views(self) -> None:
        return
