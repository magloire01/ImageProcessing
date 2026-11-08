"""Pixel accounting, optional ground area, and a comparison of the three methods.

The chart is in percent. Square meters appear only after the user turns
calibration on and supplies a pixel width and height. The sentence under
the table states which of those two readings is on screen.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.metrics import METHOD_LABELS, METHOD_ORDER
from watervision.theme import COLORS, FONT_BODY, FONT_SMALL
from watervision.ui.charts import ChartPanel
from watervision.ui.widgets import DataTable, Hint, KpiCard, SectionLabel, SwitchRow, fmt_int, fmt_pct, primary_button


class EstimationPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_columnconfigure(1, weight=1)

        self.coverage = KpiCard(left, "Water coverage")
        self.coverage.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self.pixels = KpiCard(left, "Water pixels")
        self.pixels.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self.total = KpiCard(left, "Valid pixels")
        self.total.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self.regions = KpiCard(left, "Regions / largest")
        self.regions.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))

        calib = ctk.CTkFrame(
            left, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        calib.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        SectionLabel(calib, "Pixel size").pack(anchor="w", padx=14, pady=(12, 4))
        Hint(
            calib,
            "Width and height are the ground distance represented by one pixel. Leave calibration off when that distance is unknown. The percentage does not need it.",
        ).pack(anchor="w", padx=14, pady=(0, 6))
        self.use = SwitchRow(
            calib,
            "Use this pixel size for a physical area",
            bool(app.session.params["use_calibration"]),
            lambda _value: None,
        )
        self.use.pack(anchor="w", padx=14)
        form = ctk.CTkFrame(calib, fg_color="transparent")
        form.pack(fill="x", padx=14, pady=6)
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="Width (m)", font=FONT_SMALL, text_color=COLORS["muted"]).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(form, text="Height (m)", font=FONT_SMALL, text_color=COLORS["muted"]).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.width_entry = ctk.CTkEntry(form, fg_color=COLORS["stage"], border_color=COLORS["border"], text_color=COLORS["text"])
        self.height_entry = ctk.CTkEntry(form, fg_color=COLORS["stage"], border_color=COLORS["border"], text_color=COLORS["text"])
        self.width_entry.grid(row=1, column=0, sticky="ew", pady=4)
        self.height_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.width_entry.insert(0, str(app.session.params["pixel_width_m"]))
        self.height_entry.insert(0, str(app.session.params["pixel_height_m"]))
        primary_button(calib, "Apply pixel size", self._apply).pack(fill="x", padx=14, pady=(4, 14))

        self.note = ctk.CTkLabel(
            left, text="", font=FONT_BODY, text_color=COLORS["warning"], wraplength=420, justify="left", anchor="w"
        )
        self.note.grid(row=3, column=0, columnspan=2, sticky="ew")

        self.table = DataTable(self)
        self.table.grid(row=0, column=1, sticky="ew")
        self.chart = ChartPanel(self, "Water coverage by method")
        self.chart.grid(row=1, column=1, sticky="nsew", pady=(12, 0))
        self.chart.show_message("Run segmentation to compare the three methods.")

    def sync_from_state(self) -> None:
        params = self.app.session.params
        self.use.set_value(bool(params["use_calibration"]))
        self.width_entry.delete(0, "end")
        self.height_entry.delete(0, "end")
        self.width_entry.insert(0, str(params["pixel_width_m"]))
        self.height_entry.insert(0, str(params["pixel_height_m"]))

    def refresh_views(self) -> None:
        state = self.app.session
        method = str(state.params.get("active_method", "hybrid"))
        metrics = state.metrics.get(method)
        if metrics is None:
            self.coverage.set("—")
            self.pixels.set("—")
            self.total.set("—" if not state.has_image() else fmt_int(state.height * state.width), "image pixels")
            self.regions.set("—")
            self.note.configure(
                text="Run segmentation. The active method on the segmentation page is the one summarized in the cards.",
                text_color=COLORS["muted"],
            )
        else:
            self.coverage.set(fmt_pct(metrics["percentage"]), METHOD_LABELS[method])
            self.pixels.set(fmt_int(metrics["water_pixels"]))
            self.total.set(fmt_int(metrics["total_pixels"]), "all pixels are valid")
            self.regions.set(f"{metrics['regions']} / {fmt_int(metrics['largest_region'])}", "count / largest pixels")
            physical = metrics["physical_area_m2"]
            self.note.configure(
                text=metrics["note"] if physical is None else f"{physical:,.2f} m². {metrics['note']}",
                text_color=COLORS["warning"] if physical is None else COLORS["success"],
            )
        rows = []
        percents = []
        labels = []
        for name in METHOD_ORDER:
            row = state.metrics.get(name)
            if not row:
                continue
            physical = row["physical_area_m2"]
            rows.append(
                [
                    METHOD_LABELS[name],
                    fmt_pct(row["percentage"]),
                    fmt_int(row["water_pixels"]),
                    str(row["regions"]),
                    "—" if physical is None else f"{physical:,.1f} m²",
                ]
            )
            labels.append(METHOD_LABELS[name])
            percents.append(row["percentage"])
        self.table.set_rows(["Method", "Coverage", "Water px", "Regions", "Physical"], rows)
        if percents:
            self.chart.simple_bars(labels, percents, "Water coverage by method", "Percent")
        else:
            self.chart.show_message("Run segmentation to compare the three methods.")

    def _apply(self) -> None:
        self.app.set_calibration(self.use.get(), self.width_entry.get(), self.height_entry.get())
