"""Opening view: the photograph, the active mask, and the headline numbers."""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.io_utils import overlay_mask
from watervision.processing.metrics import METHOD_LABELS
from watervision.theme import COLORS, FONT_BODY, FONT_SMALL
from watervision.ui.widgets import ImageCard, KpiCard, fmt_int, fmt_pct, primary_button, secondary_button


class DashboardPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.cards = {}
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for index, (key, title) in enumerate(
            (
                ("coverage", "Water coverage"),
                ("pixels", "Water pixels"),
                ("regions", "Detected regions"),
                ("largest", "Largest region"),
                ("time", "Processing time"),
            )
        ):
            row.grid_columnconfigure(index, weight=1)
            card = KpiCard(row, title)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            self.cards[key] = card

        self.banner = ctk.CTkLabel(
            self,
            text="Load an image or the demonstration scene to begin.",
            font=FONT_BODY,
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
        )
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        stage = ctk.CTkFrame(self, fg_color="transparent")
        stage.grid(row=2, column=0, sticky="nsew")
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_columnconfigure(1, weight=1)
        stage.grid_rowconfigure(0, weight=1)
        self.original_card = ImageCard(stage, "Original image")
        self.original_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.result_card = ImageCard(stage, "Water on black", "Contour on the water, the rest black")
        self.result_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="w", pady=(12, 0))
        primary_button(actions, "Load image", app.open_image).pack(side="left")
        secondary_button(actions, "Demonstration scene", app.load_demo).pack(side="left", padx=8)

    def sync_from_state(self) -> None:
        return

    def refresh_views(self) -> None:
        state = self.app.session
        self.original_card.show(state.original)
        method = str(state.params.get("active_method", "hybrid"))
        label = METHOD_LABELS.get(method, method)
        mask = state.active_mask()
        metrics = state.metrics.get(method)
        if state.working is not None and mask is not None:
            self.result_card.show(overlay_mask(state.working, mask))
            self.result_card.set_subtitle(f"{label} · contour on water")
        else:
            self.result_card.show(None)
            self.result_card.set_subtitle("")

        if metrics is None:
            for card in self.cards.values():
                card.set("—")
            if state.has_image():
                self.banner.configure(
                    text=(
                        f"{state.width} × {state.height} pixels loaded. "
                        "Run segmentation to estimate the water surface. "
                        "Until a pixel size is entered, results stay in pixels and percent."
                    ),
                    text_color=COLORS["muted"],
                )
            else:
                self.banner.configure(
                    text="Load an image or the demonstration scene to begin.",
                    text_color=COLORS["muted"],
                )
            return

        self.cards["coverage"].set(fmt_pct(metrics["percentage"]), f"{label} mask")
        self.cards["pixels"].set(fmt_int(metrics["water_pixels"]), f"of {fmt_int(metrics['total_pixels'])}")
        self.cards["regions"].set(str(metrics["regions"]), "8-connected")
        self.cards["largest"].set(fmt_int(metrics["largest_region"]), "pixels")
        elapsed = state.timings.get("segmentation")
        self.cards["time"].set("—" if elapsed is None else f"{elapsed:.2f} s", "segmentation")
        physical = metrics["physical_area_m2"]
        if physical is None:
            self.banner.configure(text=metrics["note"], text_color=COLORS["warning"])
        else:
            self.banner.configure(
                text=f"Physical water area ({label}): {physical:,.2f} m². {metrics['note']}",
                text_color=COLORS["success"],
            )
        # The small font on the banner keeps a long calibration sentence readable.
        self.banner.configure(font=FONT_SMALL)
