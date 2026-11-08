"""Reusable interface pieces: image previews, sliders, tables, buttons.

Image previews keep the source aspect ratio. A label in CustomTkinter shows
a CTkImage at an explicit pixel size, so the fit is computed here and the
widget is not allowed to stretch the raster.

The acquisition view is a Tk canvas because a fitted label cannot pan once
the user zooms past the available area.
"""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from watervision.theme import COLORS, FONT_BODY, FONT_BUTTON, FONT_CARD, FONT_KPI, FONT_SMALL
from watervision.ui.tooltip import Tooltip


def array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a display array to a Pillow image.

    Accepted inputs are uint8 RGB, uint8 gray, or float images in [0, 1].
    """
    data = np.asarray(array)
    if data.dtype != np.uint8:
        data = np.clip(np.round(np.asarray(data, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if data.ndim == 2:
        return Image.fromarray(data, mode="L")
    return Image.fromarray(data, mode="RGB")


def fit_box(src_w: int, src_h: int, box_w: int, box_h: int, multiplier: float = 1.0) -> tuple[int, int]:
    """Largest size that fits the box at the given zoom, preserving aspect."""
    if src_w < 1 or src_h < 1:
        return 1, 1
    scale = min(box_w / src_w, box_h / src_h) * float(multiplier)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def primary_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        fg_color=COLORS["blue"],
        hover_color=COLORS["blue_hover"],
        text_color="#FFFFFF",
        font=FONT_BUTTON,
        height=36,
        corner_radius=8,
    )


def secondary_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        fg_color="transparent",
        hover_color="#16324F",
        border_width=1,
        border_color=COLORS["accent"],
        text_color=COLORS["accent"],
        font=FONT_BUTTON,
        height=36,
        corner_radius=8,
    )


def danger_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        fg_color="transparent",
        hover_color="#3F1D24",
        border_width=1,
        border_color=COLORS["danger"],
        text_color=COLORS["danger"],
        font=FONT_BUTTON,
        height=36,
        corner_radius=8,
    )


class ImageCard(ctk.CTkFrame):
    """Titled preview that letterboxes its image inside the available area."""

    def __init__(self, master, title: str, empty_text: str = "No image loaded"):
        super().__init__(
            master,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.empty_text = empty_text
        self._array: np.ndarray | None = None
        self._photo: ctk.CTkImage | None = None
        self._token = 0
        self._drawn: tuple | None = None
        self._job = None
        # A label must never be asked to redraw while its current PhotoImage
        # has already been deleted. This 1×1 image is the stand-in used when
        # the card has nothing to show, and it is never discarded.
        blank = Image.new("RGB", (1, 1), (10, 16, 28))
        self._blank = ctk.CTkImage(light_image=blank, dark_image=blank, size=(1, 1))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=title, font=FONT_CARD, text_color=COLORS["text"], anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        self.subtitle = ctk.CTkLabel(header, text="", font=FONT_SMALL, text_color=COLORS["muted"], anchor="e")
        self.subtitle.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.view = ctk.CTkLabel(
            self,
            text=empty_text,
            text_color=COLORS["muted"],
            fg_color=COLORS["stage"],
            corner_radius=8,
            font=FONT_SMALL,
        )
        self.view.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.bind("<Configure>", self._schedule)

    def set_subtitle(self, text: str) -> None:
        self.subtitle.configure(text=text)

    def show(self, array: np.ndarray | None) -> None:
        self._array = None if array is None else np.asarray(array)
        self._token += 1
        self._drawn = None
        self._render()

    def _schedule(self, _event=None) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(40, self._render)

    def _install(self, photo: ctk.CTkImage, text: str) -> None:
        """Point the label at `photo` without deleting the image it still shows.

        CustomTkinter applies the text before the image. If the previous
        CTkImage is dropped first, Tk destroys the PhotoImage (pyimageN)
        and the text update raises « image "pyimageN" doesn't exist ».
        The previous object is held until the label has switched.
        """
        previous = self._photo
        self._photo = photo
        self.view.configure(image=photo)
        self.view.configure(text=text)
        del previous

    def _render(self) -> None:
        self._job = None
        if self._array is None:
            self._install(self._blank, self.empty_text)
            self._drawn = None
            return
        width = self.winfo_width() - 24
        height = self.winfo_height() - 52
        if width < 40 or height < 40:
            return
        key = (self._token, width, height)
        if key == self._drawn:
            return
        src_h, src_w = self._array.shape[:2]
        fitted = fit_box(src_w, src_h, width, height, 1.0)
        pil = array_to_pil(self._array).resize(fitted, Image.Resampling.LANCZOS)
        self._install(ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size), "")
        self._drawn = key


class ZoomView(ctk.CTkFrame):
    """Scrollable acquisition view.

    Multiplier 1 means "fit the current canvas". Larger multipliers zoom in
    and enable panning by dragging. The raster is never stretched: the scale
    is a single number applied to both axes.
    """

    def __init__(self, master):
        super().__init__(master, fg_color=COLORS["stage"], corner_radius=12)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=COLORS["stage"], highlightthickness=0, bd=0)
        self.xbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.ybar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.xbar.set, yscrollcommand=self.ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._array: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self.multiplier = 1.0
        self.on_zoom_changed = None
        self._job = None
        self.canvas.bind("<Configure>", self._schedule)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-1>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))

    def show(self, array: np.ndarray | None) -> None:
        self._array = None if array is None else np.asarray(array)
        self.multiplier = 1.0
        self._render()

    def set_multiplier(self, value: float) -> None:
        self.multiplier = float(np.clip(value, 1.0, 8.0))
        self._render()

    def _schedule(self, _event=None) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(40, self._render)

    def _wheel(self, event) -> str:
        if self._array is None or event.delta == 0:
            return "break"
        factor = 1.1 if event.delta > 0 else 1.0 / 1.1
        self.set_multiplier(self.multiplier * factor)
        if self.on_zoom_changed is not None:
            self.on_zoom_changed(self.multiplier)
        return "break"

    def _render(self) -> None:
        self._job = None
        self.canvas.delete("all")
        if self._array is None:
            self._photo = None
            self.xbar.grid_remove()
            self.ybar.grid_remove()
            self.canvas.create_text(
                max(self.canvas.winfo_width(), 2) // 2,
                max(self.canvas.winfo_height(), 2) // 2,
                text="Load an image to begin",
                fill=COLORS["muted"],
                font=FONT_BODY,
            )
            return
        box_w = max(self.canvas.winfo_width(), 2)
        box_h = max(self.canvas.winfo_height(), 2)
        src_h, src_w = self._array.shape[:2]
        out_w, out_h = fit_box(src_w, src_h, box_w, box_h, self.multiplier)
        pil = array_to_pil(self._array).resize((out_w, out_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        if out_w <= box_w and out_h <= box_h:
            self.canvas.create_image(box_w // 2, box_h // 2, image=self._photo, anchor="center")
            self.canvas.configure(scrollregion=(0, 0, box_w, box_h))
            self.xbar.grid_remove()
            self.ybar.grid_remove()
        else:
            self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
            self.canvas.configure(scrollregion=(0, 0, out_w, out_h))
            self.ybar.grid(row=0, column=1, sticky="ns")
            self.xbar.grid(row=1, column=0, sticky="ew")


class SliderRow(ctk.CTkFrame):
    """Label, live value, and slider. Writes happen through `command`."""

    def __init__(
        self,
        master,
        label: str,
        from_: float,
        to: float,
        initial: float,
        steps: int,
        formatter,
        command,
        tooltip: str = "",
    ):
        super().__init__(master, fg_color="transparent")
        self.formatter = formatter
        self.command = command
        self._silent = False
        self.grid_columnconfigure(0, weight=1)
        self.caption = ctk.CTkLabel(self, text=label, font=FONT_SMALL, text_color=COLORS["muted"], anchor="w")
        self.caption.grid(row=0, column=0, sticky="w")
        self.value_label = ctk.CTkLabel(self, text=formatter(initial), font=FONT_SMALL, text_color=COLORS["accent"])
        self.value_label.grid(row=0, column=1, sticky="e")
        self.slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            number_of_steps=max(int(steps), 1),
            command=self._on_slide,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent"],
            button_hover_color="#67E8F9",
            fg_color=COLORS["border"],
        )
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        self._silent = True
        self.slider.set(initial)
        self._silent = False
        if tooltip:
            Tooltip(self.caption, tooltip)

    def get(self) -> float:
        return float(self.slider.get())

    def set_value(self, value: float) -> None:
        self._silent = True
        self.slider.set(float(value))
        self.value_label.configure(text=self.formatter(float(value)))
        self._silent = False

    def _on_slide(self, value: float) -> None:
        self.value_label.configure(text=self.formatter(float(value)))
        if self._silent:
            return
        self.command(float(value))


class SwitchRow(ctk.CTkFrame):
    """Boolean parameter with a tooltip on its label."""

    def __init__(self, master, label: str, initial: bool, command, tooltip: str = ""):
        super().__init__(master, fg_color="transparent")
        self.command = command
        self._silent = False
        self.switch = ctk.CTkSwitch(
            self,
            text=label,
            command=self._on_toggle,
            font=FONT_SMALL,
            text_color=COLORS["text"],
            progress_color=COLORS["accent"],
            button_color=COLORS["text"],
            fg_color=COLORS["border"],
        )
        self.switch.pack(anchor="w", pady=4)
        self._silent = True
        if initial:
            self.switch.select()
        else:
            self.switch.deselect()
        self._silent = False
        if tooltip:
            Tooltip(self.switch, tooltip)

    def get(self) -> bool:
        return bool(self.switch.get())

    def set_value(self, value: bool) -> None:
        self._silent = True
        if value:
            self.switch.select()
        else:
            self.switch.deselect()
        self._silent = False

    def _on_toggle(self) -> None:
        if self._silent:
            return
        self.command(bool(self.switch.get()))


class KpiCard(ctk.CTkFrame):
    """One number on the dashboard or the estimation page."""

    def __init__(self, master, title: str):
        super().__init__(
            master,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        ctk.CTkLabel(self, text=title, font=FONT_SMALL, text_color=COLORS["muted"], anchor="w").pack(
            anchor="w", padx=14, pady=(12, 0)
        )
        self.value_label = ctk.CTkLabel(self, text="—", font=FONT_KPI, text_color=COLORS["text"], anchor="w")
        self.value_label.pack(anchor="w", padx=14, pady=(2, 0))
        self.note_label = ctk.CTkLabel(self, text="", font=FONT_SMALL, text_color=COLORS["muted"], anchor="w")
        self.note_label.pack(anchor="w", padx=14, pady=(0, 12))

    def set(self, value: str, note: str = "") -> None:
        self.value_label.configure(text=value)
        self.note_label.configure(text=note)


class DataTable(ctk.CTkFrame):
    """Small text table. Rebuilt on each update; the tables here are tiny."""

    def __init__(self, master):
        super().__init__(
            master,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )

    def set_rows(self, columns: list[str], rows: list[list[str]]) -> None:
        for child in self.winfo_children():
            child.destroy()
        for column, name in enumerate(columns):
            label = ctk.CTkLabel(self, text=name, font=FONT_CARD, text_color=COLORS["accent"], anchor="w")
            label.grid(row=0, column=column, sticky="ew", padx=12, pady=(10, 4))
            self.grid_columnconfigure(column, weight=1)
        for row_index, row in enumerate(rows, start=1):
            for column, value in enumerate(row):
                label = ctk.CTkLabel(self, text=value, font=FONT_BODY, text_color=COLORS["text"], anchor="w")
                label.grid(row=row_index, column=column, sticky="ew", padx=12, pady=3)
        pad = ctk.CTkLabel(self, text="")
        pad.grid(row=len(rows) + 1, column=0, pady=4)


class SectionLabel(ctk.CTkLabel):
    def __init__(self, master, text: str):
        super().__init__(master, text=text, font=FONT_CARD, text_color=COLORS["text"], anchor="w")


class Hint(ctk.CTkLabel):
    def __init__(self, master, text: str):
        super().__init__(
            master,
            text=text,
            font=FONT_SMALL,
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=360,
        )


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"

