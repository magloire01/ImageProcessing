"""Matplotlib charts embedded in the dark interface.

Figures are created from matplotlib.figure.Figure, not from pyplot, so the
GUI does not depend on pyplot's global state and a background thread that
builds a PDF cannot disturb these canvases. All drawing happens on the
Tk thread.
"""

from __future__ import annotations

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from watervision.theme import COLORS, FONT_CARD

SERIES_COLORS = {
    "HSV": "#3B82F6",
    "Fourier": "#A78BFA",
    "Hybrid": "#22D3EE",
}


class ChartPanel(ctk.CTkFrame):
    """One figure. Callers replace its contents; the canvas stays put."""

    def __init__(self, master, title: str = ""):
        super().__init__(
            master,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        if title:
            ctk.CTkLabel(self, text=title, font=FONT_CARD, text_color=COLORS["text"], anchor="w").pack(
                anchor="w", padx=12, pady=(10, 0)
            )
        self.figure = Figure(figsize=(5.2, 2.6), dpi=100, facecolor=COLORS["card"])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=COLORS["card"])
        widget.pack(fill="both", expand=True, padx=8, pady=8)

    def _axis(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor(COLORS["stage"])
        axis.tick_params(colors=COLORS["muted"], labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(COLORS["border"])
        axis.title.set_color(COLORS["text"])
        axis.xaxis.label.set_color(COLORS["muted"])
        axis.yaxis.label.set_color(COLORS["muted"])
        return axis

    def show_message(self, message: str) -> None:
        axis = self._axis()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["muted"], fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def histogram(self, centers: np.ndarray, counts: np.ndarray, title: str) -> None:
        """Luma histogram. Counts are drawn as a bar so bin width is visible."""
        axis = self._axis()
        axis.bar(centers, counts, width=centers[1] - centers[0] if len(centers) > 1 else 0.02, color=COLORS["accent"])
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_xlabel("Luma")
        axis.set_xlim(0.0, 1.0)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def simple_bars(self, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
        """One bar per label. Known method names keep their dashboard colors."""
        axis = self._axis()
        if not labels:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(0.5, 0.5, "No data", ha="center", va="center", color=COLORS["muted"])
            self.canvas.draw_idle()
            return
        colors = [SERIES_COLORS.get(label, COLORS["blue"]) for label in labels]
        axis.bar(labels, values, color=colors)
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_ylabel(ylabel)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def grouped_bars(self, categories: list[str], series: dict[str, list[float]], title: str, ylabel: str) -> None:
        """Grouped bar chart. `series` maps a legend name to one value per category."""
        axis = self._axis()
        if not categories or not series:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(0.5, 0.5, "No data", ha="center", va="center", color=COLORS["muted"])
            self.canvas.draw_idle()
            return
        x = np.arange(len(categories), dtype=float)
        width = min(0.8 / max(len(series), 1), 0.28)
        offset0 = (len(series) - 1) / 2.0
        for index, (name, values) in enumerate(series.items()):
            axis.bar(
                x + (index - offset0) * width,
                values,
                width=width * 0.92,
                label=name,
                color=SERIES_COLORS.get(name, COLORS["blue"]),
            )
        axis.set_xticks(x, categories)
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_ylabel(ylabel)
        axis.legend(
            facecolor=COLORS["card"],
            edgecolor=COLORS["border"],
            labelcolor=COLORS["text"],
            fontsize=8,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()
