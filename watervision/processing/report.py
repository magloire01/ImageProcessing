"""CSV table and multi-page PDF report.

The PDF is drawn with Matplotlib's PDF backend. No extra document library
is required. Figures are created off the screen, so this module is safe to
call from a background thread.

Pages, when the corresponding data exist:

1. Title, file, image size, and the question the report answers.
2. Working image and the active-method overlay.
3. The three cleaned masks.
4. Area table, with an explicit pixel-based or calibrated label.
5. Fourier spectrum and filtered image, if they were computed.
6. Ground-truth scores, if a reference mask was loaded.
7. Parameters and a short statement of what each method does and does not claim.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from watervision import __version__
from watervision.processing.io_utils import mask_to_rgb, overlay_mask
from watervision.processing.metrics import METHOD_LABELS, METHOD_ORDER


def _format_area(metrics: dict) -> str:
    physical = metrics.get("physical_area_m2")
    if physical is None:
        return "not calibrated"
    return f"{physical:,.2f} m²"


def write_csv(path: str, state_view: dict) -> None:
    """Write metrics, evaluation, and parameters into one UTF-8 CSV file.

    The file has three labeled sections so a spreadsheet can read it without
    a second export. UTF-8 with a BOM is used so Excel on Windows detects the
    encoding.
    """
    metrics: dict = state_view.get("metrics") or {}
    evaluation: dict = state_view.get("evaluation") or {}
    params: dict = state_view.get("params") or {}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["WaterVision results", __version__])
        writer.writerow(["file", state_view.get("filename", "")])
        writer.writerow(["generated", datetime.now().isoformat(timespec="seconds")])
        writer.writerow([])
        writer.writerow(
            [
                "method",
                "water_pixels",
                "total_pixels",
                "water_percentage",
                "regions",
                "largest_region_pixels",
                "physical_area_m2",
                "area_status",
            ]
        )
        for name in METHOD_ORDER:
            row = metrics.get(name)
            if not row:
                continue
            physical = row.get("physical_area_m2")
            writer.writerow(
                [
                    METHOD_LABELS[name],
                    row["water_pixels"],
                    row["total_pixels"],
                    f"{row['percentage']:.6f}",
                    row["regions"],
                    row["largest_region"],
                    "" if physical is None else f"{physical:.6f}",
                    "pixel-based estimate" if physical is None else "calibrated",
                ]
            )
        writer.writerow([])
        writer.writerow(["evaluation"])
        writer.writerow(["method", "iou", "precision", "recall", "f1", "pixel_accuracy"])
        if not evaluation:
            writer.writerow(["none", "", "", "", "", ""])
        for name in METHOD_ORDER:
            row = evaluation.get(name)
            if not row:
                continue
            writer.writerow(
                [
                    METHOD_LABELS[name],
                    f"{row['iou']:.6f}",
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1']:.6f}",
                    f"{row['accuracy']:.6f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["parameter", "value"])
        for key in sorted(params):
            writer.writerow([key, params[key]])


def _style(figure: Figure):
    figure.patch.set_facecolor("#0B1220")
    return figure


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    figure = _style(Figure(figsize=(8.27, 11.69)))
    figure.suptitle(title, color="#E8F1FF", fontsize=16, y=0.97)
    body = "\n".join(lines)
    figure.text(0.08, 0.92, body, va="top", ha="left", color="#D6E4F5", fontsize=10, family="DejaVu Sans")
    pdf.savefig(figure)
    figure.clear()


def _add_image_page(pdf: PdfPages, title: str, images: list[tuple[str, np.ndarray]]) -> None:
    count = len(images)
    figure = _style(Figure(figsize=(8.27, 11.69)))
    figure.suptitle(title, color="#E8F1FF", fontsize=14, y=0.98)
    for index, (caption, image) in enumerate(images, start=1):
        axis = figure.add_subplot(count, 1, index)
        axis.imshow(np.asarray(image))
        axis.set_title(caption, color="#8BA3C7", fontsize=10)
        axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure)
    figure.clear()


def write_pdf(path: str, state_view: dict) -> None:
    """Build the academic report from the current workspace snapshot.

    `state_view` is a plain dictionary of arrays and numbers copied on the
    calling side. This function does not touch the live application state,
    so it can run while the user keeps working only if the caller has already
    copied what it needs. The interface waits for it; the copy still keeps
    this module free of any widget dependency.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = state_view.get("working")
    if rgb is None:
        raise ValueError("Load an image before exporting a report.")

    filename = state_view.get("filename") or "untitled"
    params = state_view.get("params") or {}
    metrics = state_view.get("metrics") or {}
    evaluation = state_view.get("evaluation") or {}
    active = str(params.get("active_method", "hybrid"))
    height, width = rgb.shape[:2]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    with PdfPages(destination) as pdf:
        info = pdf.infodict()
        info["Title"] = f"WaterVision report — {filename}"
        info["Author"] = f"WaterVision {__version__}"
        info["Subject"] = "Natural water surface detection"

        area_lines = []
        for name in METHOD_ORDER:
            row = metrics.get(name)
            if not row:
                continue
            area_lines.append(
                f"{METHOD_LABELS[name]:<10}  {row['percentage']:6.2f}%   "
                f"{row['water_pixels']:,} px   {row['regions']} regions   "
                f"largest {row['largest_region']:,} px   {_format_area(row)}"
            )
        if not area_lines:
            area_lines.append("Segmentation has not been run. No area is reported.")
        active_metrics = metrics.get(active)
        if active_metrics:
            area_lines.append("")
            area_lines.append(active_metrics["note"])

        _add_text_page(
            pdf,
            "WaterVision — analysis report",
            [
                f"Generated {now}    software {__version__}",
                f"Image: {filename}",
                f"Working resolution: {width} × {height} pixels   ({width * height:,} valid pixels)",
                f"Active method for the highlighted figure: {METHOD_LABELS.get(active, active)}",
                "",
                "This report describes a pixel classification of one photograph.",
                "Water percentage = (water pixels / image pixels) × 100.",
                "A value in square meters is included only when a pixel size was entered",
                "and calibration was turned on. Otherwise every area statement is a pixel",
                "count or a percentage, not a ground area.",
                "",
                "Algorithms: NumPy and SciPy. OpenCV is not used.",
                "",
                "Coverage",
                *area_lines,
            ],
        )

        overlay = None
        masks = state_view.get("masks") or {}
        if active in masks:
            overlay = overlay_mask(rgb, masks[active])
        pictures = [("Working image", rgb)]
        if overlay is not None:
            pictures.append((f"{METHOD_LABELS.get(active, active)} water kept, rest black, white contour", overlay))
        _add_image_page(pdf, "Input and highlighted water", pictures)

        mask_pictures = []
        for name in METHOD_ORDER:
            if name in masks:
                mask_pictures.append((f"{METHOD_LABELS[name]} mask", mask_to_rgb(masks[name])))
        if mask_pictures:
            _add_image_page(pdf, "Segmentation masks after morphology", mask_pictures)
        else:
            _add_text_page(
                pdf,
                "Segmentation masks",
                ["Segmentation has not been run for this workspace."],
            )

        spectrum = state_view.get("spectrum_rgb")
        filtered = state_view.get("filtered_rgb")
        if spectrum is not None and filtered is not None:
            _add_image_page(
                pdf,
                "Fourier analysis",
                [
                    ("Log magnitude spectrum (display stretch only)", spectrum),
                    (state_view.get("filter_label") or "Filtered reconstruction", filtered),
                ],
            )

        if evaluation:
            lines = ["Ground truth: " + (state_view.get("ground_truth_name") or "reference mask")]
            if state_view.get("ground_truth_resized"):
                lines.append("The reference was resized with nearest-neighbor to the working image.")
            lines.append("")
            lines.append(f"{'method':<10}{'IoU':>8}{'Prec.':>8}{'Recall':>8}{'F1':>8}{'Acc.':>8}")
            for name in METHOD_ORDER:
                row = evaluation.get(name)
                if not row:
                    continue
                lines.append(
                    f"{METHOD_LABELS[name]:<10}{row['iou']:8.3f}{row['precision']:8.3f}"
                    f"{row['recall']:8.3f}{row['f1']:8.3f}{row['accuracy']:8.3f}"
                )
            lines.append("")
            lines.append(
                "F1 equals the Dice coefficient for one binary class. "
                "Pixel accuracy includes the true negatives and stays high when most of the "
                "frame is dry, even if the water outline is poor. IoU and F1 describe the water region."
            )
            _add_text_page(pdf, "Evaluation against a reference mask", lines)

        method_notes = [
            "Method 1 — HSV. A pixel is water when hue, saturation, and value",
            "lie in the selected intervals. A second hue band and a dark-water",
            "rule can be enabled. The test does not look at texture.",
            "",
            "Method 2 — Fourier. Local high-frequency energy is the share of",
            "windowed-patch spectral energy outside a low-frequency disk.",
            "Pixels below the texture threshold are smooth candidates.",
            "Frequency information alone is not treated as proof of water:",
            "sand, pavement, and shadow are smooth as well.",
            "",
            "Method 3 — Hybrid. A score S = wc C + wt T + wh H is thresholded.",
            "C is soft HSV membership, T is soft smoothness, H is local",
            "homogeneity. Weights are normalized to sum to 1. This is the",
            "method intended for the reported surface.",
            "",
            "Morphology, in order: opening, closing, hole filling, removal of",
            "components smaller than the selected pixel count. Opening deletes",
            "thin false positives before closing can connect them.",
            "",
            "Parameters",
        ]
        for key in sorted(params):
            method_notes.append(f"  {key}: {params[key]}")
        _add_text_page(pdf, "Methods and parameters", method_notes)
