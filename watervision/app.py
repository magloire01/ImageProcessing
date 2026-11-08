"""WaterVision application window.

Heavy numerical work runs on a daemon thread. Widget updates are scheduled
back onto the Tk thread with `after`. A second request is refused while a
task holds the busy flag, so two segmentations cannot write the masks at
once. Slider updates that only recolor a cached spectrum use a separate
generation counter and do not take the busy flag.
"""

from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np

from watervision import __version__
from watervision.processing.color import rgb_to_gray
from watervision.processing.demo import make_demo_scene
from watervision.processing.fourier import analyze_spectrum, fft2_shifted, filter_cached
from watervision.processing.fourier import local_high_frequency_ratio
from watervision.processing.io_utils import (
    ImageIOError,
    load_mask,
    load_rgb,
    mask_to_label_image,
    overlay_mask,
    save_rgb,
)
from watervision.processing.metrics import estimate_all, evaluate_all, resize_nearest
from watervision.processing.report import write_csv, write_pdf
from watervision.processing.segmentation import segment
from watervision.state import DEFAULT_PARAMS, PREPROCESS_KEYS, SEGMENT_KEYS, AppState
from watervision.theme import COLORS
from watervision.ui.pages import (
    AcquisitionPage,
    DashboardPage,
    EstimationPage,
    EvaluationPage,
    ExportPage,
    FourierPage,
    PreprocessingPage,
    SegmentationPage,
    SettingsPage,
)
from watervision.ui.shell import Header, Sidebar

# Windows file dialogs split patterns with semicolons. Spaces hide every file.
IMAGE_TYPES = [
    ("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg;*.jpeg"),
    ("TIFF", "*.tif;*.tiff"),
    ("BMP", "*.bmp"),
    ("All files", "*.*"),
]


def _positive_length(text: str) -> float | None:
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


class WaterVisionApp(ctk.CTk):
    """Desktop shell. Pages call back into this object; they do not own threads."""

    def __init__(self):
        super().__init__()
        self.title(f"WaterVision — Natural Water Surface Detection  {__version__}")
        self.geometry("1440x900")
        self.minsize(1180, 740)
        self.configure(fg_color=COLORS["bg"])
        self.session = AppState()
        self.busy = False
        self.current = "dashboard"
        self._filter_generation = 0
        self._filter_job = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, self.navigate)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        main = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        self.header = Header(main)
        self.header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))

        self.content = ctk.CTkFrame(main, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 16))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages = {
            "dashboard": DashboardPage(self.content, self),
            "acquisition": AcquisitionPage(self.content, self),
            "preprocessing": PreprocessingPage(self.content, self),
            "fourier": FourierPage(self.content, self),
            "segmentation": SegmentationPage(self.content, self),
            "estimation": EstimationPage(self.content, self),
            "evaluation": EvaluationPage(self.content, self),
            "export": ExportPage(self.content, self),
            "settings": SettingsPage(self.content, self),
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self.bind("<Control-o>", lambda _event: self.open_image())
        self.navigate("dashboard")

    # ------------------------------------------------------------------
    # Threading and navigation
    # ------------------------------------------------------------------

    def spawn(self, func, on_success, on_error=None) -> None:
        """Run `func` off the UI thread and deliver the result with `after`."""
        import threading

        def work():
            try:
                result = func()
                error = None
            except Exception as exc:  # shown in a dialog; also printed for the lab log
                result = None
                error = exc

            def finish():
                try:
                    if error is not None:
                        if on_error is not None:
                            on_error(error)
                        else:
                            self._show_error(error)
                    else:
                        on_success(result)
                except Exception as callback_error:
                    self._show_error(callback_error)

            self.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def run_task(self, status: str, func, on_success) -> bool:
        if self.busy:
            self.session.status = "A computation is already running."
            self._render_header()
            messagebox.showinfo(
                "WaterVision",
                "A computation is still running. Wait until it finishes, then try again. "
                "The picture on screen stays unchanged until then.",
            )
            return False
        self.busy = True
        self.session.status = status
        self._render_header()

        def done(result):
            self.busy = False
            on_success(result)

        def failed(exc: Exception):
            self.busy = False
            self.session.status = "The operation failed."
            self._render_header()
            self._show_error(exc)

        self.spawn(func, done, failed)
        return True

    def _show_error(self, exc: BaseException) -> None:
        traceback.print_exception(exc)
        text = str(exc).strip() or exc.__class__.__name__
        messagebox.showerror("WaterVision", text)

    def navigate(self, page_id: str) -> None:
        if page_id not in self.pages:
            return
        self.current = page_id
        for key, page in self.pages.items():
            if key == page_id:
                page.grid()
            else:
                page.grid_remove()
        self.sidebar.set_active(page_id)
        page = self.pages[page_id]
        page.sync_from_state()
        page.refresh_views()
        self._render_header()
        on_enter = getattr(page, "on_enter", None)
        if on_enter is not None:
            on_enter()

    def refresh_current(self) -> None:
        page = self.pages[self.current]
        page.refresh_views()
        self._render_header()

    def _render_header(self) -> None:
        self.header.render(self.current, self.session.filename, self.session.status, self.busy)

    def require_image(self) -> bool:
        if self.session.has_image():
            return True
        messagebox.showinfo(
            "WaterVision",
            "Load an image, or open the demonstration scene, before running this step.",
        )
        return False

    def energy_key(self) -> tuple:
        params = self.session.params
        return (
            self.session.revision,
            int(params["patch_window"]),
            round(float(params["patch_cutoff"]), 4),
            int(params["max_frequency_windows"]),
        )

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def open_image(self) -> None:
        """Replace the workspace with a file chosen by the user.

        The dialog is parented to this window. Without that, on Windows it
        opens behind WaterVision and the demonstration scene stays on screen
        because no file was actually returned. A load started while a
        segmentation is running is refused, otherwise the finishing demo
        would still own the panels.
        """
        if self.busy:
            messagebox.showinfo(
                "WaterVision",
                "Wait for the current operation to finish, then click Load image again. "
                "The demonstration scene stays visible until your file is opened.",
            )
            return
        self.lift()
        self.focus_force()
        path = filedialog.askopenfilename(
            parent=self,
            title="Open your image",
            filetypes=IMAGE_TYPES,
        )
        if not path:
            return

        def work():
            return load_rgb(path)

        def done(result):
            rgb, note = result
            self.adopt_image(rgb, Path(path).name, path, note)
            self.session.status = note or f"Loaded {self.session.filename}"
            # Show the file at once, before segmentation replaces the old lake.
            self.navigate("dashboard")
            self._reveal_on_dashboard = True
            self.run_segmentation()

        self.run_task("Loading image", work, done)

    def load_demo(self) -> None:
        def work():
            return make_demo_scene()

        def done(result):
            rgb, truth = result
            self.adopt_image(
                rgb,
                "demonstration_scene.png",
                "",
                "Synthetic scene. The sandbar is smooth and dry; the grey pond is dark water.",
            )
            self.session.ground_truth = truth
            self.session.ground_truth_name = "Synthetic reference"
            self.session.ground_truth_resized = False
            self._reveal_on_dashboard = True
            self.refresh_current()
            # Segmentation starts after this callback returns, so the busy
            # flag is already clear. The reference is scored at the end.
            self.run_segmentation()

        self.run_task("Building demonstration scene", work, done)

    def adopt_image(self, rgb: np.ndarray, filename: str, path: str, note: str = "") -> None:
        self.session.set_original(rgb, filename, path, note)
        preprocessing = self.pages["preprocessing"]
        preprocessing.on_image_replaced()

    def reset_workspace(self) -> None:
        if self.session.has_image() or self.session.filename:
            agreed = messagebox.askyesno(
                "WaterVision",
                "Clear the image, masks, spectrum, and parameters?",
            )
            if not agreed:
                return
        self.session.reset_workspace()
        for page in self.pages.values():
            page.sync_from_state()
            hook = getattr(page, "on_workspace_reset", None)
            if hook is not None:
                hook()
        self.refresh_current()

    # ------------------------------------------------------------------
    # Preprocessing and Fourier
    # ------------------------------------------------------------------

    def commit_preprocessing(self, rgb: np.ndarray, params: dict) -> None:
        self.session.set_working(np.ascontiguousarray(rgb), params)
        self.session.status = "Preprocessing applied. Previous masks and the spectrum were cleared."
        self.refresh_current()

    def compute_spectrum(self) -> None:
        if not self.require_image():
            return
        image = self.session.working
        revision = self.session.revision

        def work():
            import time

            started = time.perf_counter()
            spectrum = fft2_shifted(rgb_to_gray(image))
            return spectrum, time.perf_counter() - started

        def done(payload):
            spectrum, elapsed = payload
            if revision != self.session.revision:
                self.session.status = "Fourier result discarded because the image changed."
                self.refresh_current()
                return
            self.session.spectrum = spectrum
            self.session.spectrum_revision = revision
            self.session.timings["fourier"] = elapsed
            self.session.status = f"Fourier transform computed in {elapsed:.2f} s"
            self.refresh_current()
            self.apply_filter_async()

        self.run_task("Computing the 2D Fourier transform", work, done)

    def schedule_filter(self) -> None:
        if self._filter_job is not None:
            self.after_cancel(self._filter_job)
        self._filter_job = self.after(120, self.apply_filter_async)

    def apply_filter_async(self) -> None:
        self._filter_job = None
        spectrum = self.session.spectrum
        if spectrum is None:
            return
        self._filter_generation += 1
        generation = self._filter_generation
        params = dict(self.session.params)

        def work():
            return filter_cached(spectrum, params)

        def done(result):
            if generation != self._filter_generation or self.session.spectrum is not spectrum:
                return
            self.session.spectrum_rgb = result["spectrum_rgb"]
            self.session.filtered_rgb = result["filtered_rgb"]
            self.session.filter_label = result["filter_label"]
            self.refresh_current()

        self.spawn(work, done)

    def compute_local_energy(self) -> None:
        if not self.require_image():
            return
        image = self.session.working
        params = dict(self.session.params)
        revision = self.session.revision
        key = (
            revision,
            int(params["patch_window"]),
            round(float(params["patch_cutoff"]), 4),
            int(params["max_frequency_windows"]),
        )

        def work():
            return local_high_frequency_ratio(
                rgb_to_gray(image),
                window=key[1],
                cutoff_ratio=key[2],
                max_windows=key[3],
            )

        def done(energy):
            if revision != self.session.revision:
                return
            self.session.energy = energy
            self.session.energy_key = key
            self.session.status = "Local frequency map ready. Bright regions are textured, not necessarily dry land."
            self.refresh_current()

        self.run_task("Computing local frequency energy", work, done)

    # ------------------------------------------------------------------
    # Segmentation, area, evaluation
    # ------------------------------------------------------------------

    def run_segmentation(self) -> None:
        if not self.require_image():
            return
        image = self.session.working
        params = dict(self.session.params)
        revision = self.session.revision
        key = (
            revision,
            int(params["patch_window"]),
            round(float(params["patch_cutoff"]), 4),
            int(params["max_frequency_windows"]),
        )
        energy = self.session.energy if self.session.energy_key == key else None

        def work():
            import time

            started = time.perf_counter()
            result = segment(image, params, energy=energy)
            return result, time.perf_counter() - started

        def done(payload):
            result, elapsed = payload
            if revision != self.session.revision:
                self.session.status = "Segmentation discarded because the image changed."
                self.refresh_current()
                return
            self.session.energy = result.energy
            self.session.energy_key = key
            self.session.mask_raw = result.raw
            self.session.mask_clean = result.clean
            self.session.weights_used = result.weights
            self.session.metrics = estimate_all(result.clean, self.session.params)
            self.session.timings["segmentation"] = elapsed
            self.session.last_timing_s = elapsed
            if self.session.ground_truth is not None:
                self.session.evaluation = evaluate_all(result.clean, self.session.ground_truth)
            else:
                self.session.evaluation = {}
            method = str(self.session.params.get("active_method", "hybrid"))
            coverage = self.session.metrics.get(method, {}).get("percentage")
            if coverage is None:
                self.session.status = f"Segmentation finished in {elapsed:.2f} s"
            else:
                self.session.status = f"Segmentation finished in {elapsed:.2f} s · {coverage:.2f}% water ({method})"
            if getattr(self, "_reveal_on_dashboard", False):
                self._reveal_on_dashboard = False
                self.navigate("dashboard")
            else:
                self.refresh_current()

        self.run_task("Segmenting water", work, done)

    def reset_segmentation_params(self) -> None:
        for key in SEGMENT_KEYS:
            self.session.params[key] = deepcopy(DEFAULT_PARAMS[key])
        self.session.mask_raw = {}
        self.session.mask_clean = {}
        self.session.metrics = {}
        self.session.evaluation = {}
        self.session.weights_used = {}
        self.session.energy = None
        self.session.energy_key = None
        self.pages["segmentation"].sync_from_state()
        self.pages["fourier"].sync_from_state()
        self.session.status = "Segmentation parameters restored. Run segmentation to rebuild the masks."
        self.refresh_current()

    def recompute_metrics(self) -> None:
        if not self.session.mask_clean:
            return
        self.session.metrics = estimate_all(self.session.mask_clean, self.session.params)
        if self.session.ground_truth is not None:
            self.session.evaluation = evaluate_all(self.session.mask_clean, self.session.ground_truth)

    def set_calibration(self, enabled: bool, width_text: str, height_text: str) -> None:
        if not enabled:
            self.session.params["use_calibration"] = False
            self.recompute_metrics()
            self.session.status = "Calibration off. The estimate is a pixel percentage, not a ground area."
            self.refresh_current()
            return
        width = _positive_length(width_text)
        height = _positive_length(height_text)
        if width is None or height is None:
            messagebox.showerror(
                "WaterVision",
                "Pixel width and height must be positive numbers in meters. "
                "Turn calibration off if the pixel size is unknown.",
            )
            return
        self.session.params["use_calibration"] = True
        self.session.params["pixel_width_m"] = width
        self.session.params["pixel_height_m"] = height
        self.recompute_metrics()
        self.session.status = f"Pixel size set to {width:.4g} m × {height:.4g} m."
        self.refresh_current()

    def open_ground_truth(self) -> None:
        if not self.require_image():
            return
        path = filedialog.askopenfilename(title="Open a reference mask", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            mask = load_mask(path)
        except ImageIOError as exc:
            self._show_error(exc)
            return
        resized = False
        shape = (self.session.height, self.session.width)
        if mask.shape != shape:
            mask = resize_nearest(mask, shape)
            resized = True
        if not np.any(mask):
            messagebox.showwarning(
                "WaterVision",
                "The reference mask has no water pixels after thresholding. Scores will describe an empty water class.",
            )
        self.session.ground_truth = mask
        self.session.ground_truth_name = Path(path).name
        self.session.ground_truth_resized = resized
        if self.session.mask_clean:
            self.session.evaluation = evaluate_all(self.session.mask_clean, mask)
            self.session.status = "Reference scored against the three methods."
        else:
            self.session.evaluation = {}
            self.session.status = "Reference loaded. Run segmentation to compute the scores."
        if resized:
            self.session.status += " The mask was resized with nearest neighbor."
        self.refresh_current()

    def score_ground_truth(self) -> None:
        if not self.require_image():
            return
        if self.session.ground_truth is None:
            messagebox.showinfo("WaterVision", "Load a binary reference mask first.")
            return
        if not self.session.mask_clean:
            messagebox.showinfo("WaterVision", "Run segmentation before scoring the reference.")
            return
        self.session.evaluation = evaluate_all(self.session.mask_clean, self.session.ground_truth)
        self.session.status = "Reference scored against the three methods."
        self.refresh_current()

    def restore_defaults(self) -> None:
        agreed = messagebox.askyesno(
            "WaterVision",
            "Restore every parameter and, if an image is loaded, return the working image to the original?",
        )
        if not agreed:
            return
        self.session.params = deepcopy(DEFAULT_PARAMS)
        if self.session.original is not None:
            identity = {key: self.session.params[key] for key in PREPROCESS_KEYS}
            self.session.set_working(self.session.original.copy(), identity)
            self.pages["preprocessing"].on_image_replaced()
        for page in self.pages.values():
            page.sync_from_state()
        self.session.status = "Parameters restored."
        self.refresh_current()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _selected_mask(self):
        method = self.pages["export"].selected_method()
        mask = self.session.mask_clean.get(method)
        if mask is None:
            messagebox.showinfo(
                "WaterVision",
                "Run segmentation before saving a mask or a highlighted image.",
            )
            return None, method
        return mask, method

    def export_mask(self) -> None:
        mask, method = self._selected_mask()
        if mask is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save binary mask",
            defaultextension=".png",
            initialfile=f"{method}_mask.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        label = mask_to_label_image(mask)

        def work():
            save_rgb(path, label)

        self.run_task("Saving mask", work, lambda _result: self._saved(path))

    def export_overlay(self) -> None:
        if not self.require_image():
            return
        mask, method = self._selected_mask()
        if mask is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save highlighted image",
            defaultextension=".png",
            initialfile=f"{method}_overlay.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        image = self.session.working
        painted = overlay_mask(image, mask)

        def work():
            save_rgb(path, painted)

        self.run_task("Saving highlighted image", work, lambda _result: self._saved(path))

    def export_spectrum(self) -> None:
        if not self.require_image():
            return
        path = filedialog.asksaveasfilename(
            title="Save Fourier spectrum",
            defaultextension=".png",
            initialfile="fourier_spectrum.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        image = self.session.working
        params = dict(self.session.params)
        existing = self.session.spectrum_rgb
        revision = self.session.revision

        def work():
            if existing is not None:
                picture = existing
                fresh = None
            else:
                fresh = analyze_spectrum(image, params)
                picture = fresh["spectrum_rgb"]
            save_rgb(path, picture)
            return fresh

        def done(fresh):
            if fresh is not None and revision == self.session.revision and self.session.spectrum is None:
                self.session.spectrum = fresh["spectrum"]
                self.session.spectrum_revision = revision
                self.session.spectrum_rgb = fresh["spectrum_rgb"]
                self.session.filtered_rgb = fresh["filtered_rgb"]
                self.session.filter_label = fresh["filter_label"]
            self._saved(path)

        self.run_task("Saving Fourier spectrum", work, done)

    def _export_view(self) -> dict:
        return {
            "working": self.session.working,
            "filename": self.session.filename,
            "params": dict(self.session.params),
            "metrics": deepcopy(self.session.metrics),
            "evaluation": deepcopy(self.session.evaluation),
            "masks": dict(self.session.mask_clean),
            "spectrum_rgb": self.session.spectrum_rgb,
            "filtered_rgb": self.session.filtered_rgb,
            "filter_label": self.session.filter_label,
            "ground_truth_name": self.session.ground_truth_name,
            "ground_truth_resized": self.session.ground_truth_resized,
        }

    def export_csv(self) -> None:
        if not self.require_image():
            return
        path = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".csv",
            initialfile="watervision_results.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        view = self._export_view()

        def work():
            write_csv(path, view)

        self.run_task("Writing CSV", work, lambda _result: self._saved(path))

    def export_pdf(self) -> None:
        if not self.require_image():
            return
        path = filedialog.asksaveasfilename(
            title="Export PDF report",
            defaultextension=".pdf",
            initialfile="watervision_report.pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if not path:
            return
        view = self._export_view()

        def work():
            write_pdf(path, view)

        self.run_task("Writing PDF report", work, lambda _result: self._saved(path))

    def _saved(self, path: str) -> None:
        self.session.status = f"Saved {Path(path).name}"
        self.refresh_current()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = WaterVisionApp()
    app.mainloop()
