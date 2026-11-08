"""Ground-truth scores for the three cleaned masks.

IoU, precision, recall, and F1 describe the water class. Pixel accuracy is
shown because it is a standard summary, with an explicit warning: when most
of the frame is dry, accuracy stays high even if the shoreline is wrong.
F1 for one binary class is the Dice coefficient.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision.processing.io_utils import mask_to_rgb
from watervision.processing.metrics import METHOD_LABELS, METHOD_ORDER, agreement_image
from watervision.theme import COLORS, FONT_SMALL
from watervision.ui.charts import ChartPanel
from watervision.ui.widgets import DataTable, Hint, ImageCard, primary_button


class EvaluationPage(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        primary_button(bar, "Load reference mask", app.open_ground_truth).pack(side="left")
        primary_button(bar, "Score methods", app.score_ground_truth).pack(side="left", padx=8)
        self.caption = ctk.CTkLabel(bar, text="No reference loaded.", font=FONT_SMALL, text_color=COLORS["muted"])
        self.caption.pack(side="left", padx=12)

        self.truth_card = ImageCard(self, "Reference mask", "Water is cyan")
        self.truth_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.agree_card = ImageCard(self, "Agreement with the active method", "Cyan hit, amber false water, violet missed water")
        self.agree_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        self.table = DataTable(self)
        self.table.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.chart = ChartPanel(self, "Scores by method")
        self.chart.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.grid_rowconfigure(3, weight=1)
        Hint(
            self,
            "Pixel accuracy counts dry background as true negatives, so it stays high when water is a small part of the frame. IoU and F1 describe the water region. F1 equals the Dice coefficient for a binary mask.",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.chart.show_message("Load a reference mask after segmentation.")

    def sync_from_state(self) -> None:
        return

    def refresh_views(self) -> None:
        state = self.app.session
        if state.ground_truth is None:
            self.truth_card.show(None)
            self.caption.configure(text="No reference loaded.")
        else:
            self.truth_card.show(mask_to_rgb(state.ground_truth))
            note = state.ground_truth_name or "Reference"
            if state.ground_truth_resized:
                note += "  ·  resized with nearest neighbor to the working image"
            self.caption.configure(text=note)
        method = str(state.params.get("active_method", "hybrid"))
        prediction = state.mask_clean.get(method)
        if prediction is not None and state.ground_truth is not None:
            self.agree_card.show(agreement_image(prediction, state.ground_truth))
            self.agree_card.set_subtitle(METHOD_LABELS[method])
        else:
            self.agree_card.show(None)
            self.agree_card.set_subtitle("")

        rows = []
        method_series: dict[str, list[float]] = {}
        for name in METHOD_ORDER:
            scores = state.evaluation.get(name)
            if not scores:
                continue
            rows.append(
                [
                    METHOD_LABELS[name],
                    f"{scores['iou']:.3f}",
                    f"{scores['precision']:.3f}",
                    f"{scores['recall']:.3f}",
                    f"{scores['f1']:.3f}",
                    f"{scores['accuracy']:.3f}",
                ]
            )
            method_series[METHOD_LABELS[name]] = [
                scores["iou"], scores["precision"], scores["recall"], scores["f1"]
            ]
        self.table.set_rows(["Method", "IoU", "Precision", "Recall", "F1", "Accuracy"], rows)
        if method_series:
            self.chart.grouped_bars(
                ["IoU", "Precision", "Recall", "F1"],
                method_series,
                "Segmentation scores",
                "Score",
            )
        else:
            self.chart.show_message("Load a reference mask after segmentation.")
