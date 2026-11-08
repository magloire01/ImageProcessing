"""Sidebar and header shared by every module.

The header is the only place that reports the current file and whether a
background computation is running. Pages do not each invent a status line.
"""

from __future__ import annotations

import customtkinter as ctk

from watervision import __version__
from watervision.theme import COLORS, FONT_BODY, FONT_LOGO, FONT_SMALL, FONT_TITLE

NAV_ITEMS = (
    ("dashboard", "Dashboard"),
    ("acquisition", "Image Acquisition"),
    ("preprocessing", "Preprocessing"),
    ("fourier", "Fourier Analysis"),
    ("segmentation", "Segmentation"),
    ("estimation", "Area Estimation"),
    ("evaluation", "Evaluation"),
    ("export", "Export"),
    ("settings", "Settings"),
)

PAGE_COPY = {
    "dashboard": ("Dashboard", "Overview of the loaded image and the latest water estimate"),
    "acquisition": ("Image Acquisition", "Load a photograph, inspect its size, zoom, or reset the workspace"),
    "preprocessing": ("Preprocessing", "Grayscale is derived on demand. These tools edit the RGB working image"),
    "fourier": ("Fourier Analysis", "Centered 2D spectrum, frequency filters, and local high-frequency energy"),
    "segmentation": ("Segmentation", "HSV, local frequency, and the hybrid score, then morphological cleanup"),
    "estimation": ("Area Estimation", "Pixel coverage, and a ground area only when a pixel size is provided"),
    "evaluation": ("Evaluation", "Score each method against a binary reference mask"),
    "export": ("Export", "Save masks, the highlighted image, the spectrum, a CSV table, or a PDF report"),
    "settings": ("Settings", "Analysis resolution and a short statement of what the methods claim"),
}


class Sidebar(ctk.CTkFrame):
    """Module navigation. Every button changes the visible page."""

    def __init__(self, master, on_select):
        super().__init__(master, fg_color=COLORS["sidebar"], corner_radius=0, width=232)
        self.pack_propagate(False)
        self.on_select = on_select
        self.buttons: dict[str, ctk.CTkButton] = {}

        brand = ctk.CTkFrame(self, fg_color="transparent")
        brand.pack(fill="x", padx=16, pady=(20, 8))
        mark = ctk.CTkFrame(brand, fg_color=COLORS["accent"], corner_radius=8, width=36, height=36)
        mark.pack_propagate(False)
        mark.pack(side="left")
        ctk.CTkLabel(mark, text="WV", font=FONT_LOGO, text_color=COLORS["bg"]).pack(expand=True)
        titles = ctk.CTkFrame(brand, fg_color="transparent")
        titles.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(titles, text="WATER VISION", font=FONT_BODY, text_color=COLORS["text"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(titles, text="Surface detection", font=FONT_SMALL, text_color=COLORS["muted"], anchor="w").pack(anchor="w")

        ctk.CTkFrame(self, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16, pady=(12, 12))

        for page_id, label in NAV_ITEMS:
            button = ctk.CTkButton(
                self,
                text=label,
                anchor="w",
                command=lambda selected=page_id: self.on_select(selected),
                fg_color="transparent",
                hover_color="#16263C",
                text_color=COLORS["text"],
                font=FONT_BODY,
                height=36,
                corner_radius=8,
            )
            button.pack(fill="x", padx=12, pady=2)
            self.buttons[page_id] = button

        ctk.CTkLabel(
            self,
            text=f"Academic build  {__version__}\nNumPy · SciPy · Pillow",
            font=FONT_SMALL,
            text_color=COLORS["muted"],
            justify="left",
            anchor="w",
        ).pack(side="bottom", anchor="w", padx=18, pady=16)

    def set_active(self, page_id: str) -> None:
        for key, button in self.buttons.items():
            if key == page_id:
                button.configure(fg_color=COLORS["blue"], hover_color=COLORS["blue_hover"], text_color="#FFFFFF")
            else:
                button.configure(fg_color="transparent", hover_color="#16263C", text_color=COLORS["text"])


class Header(ctk.CTkFrame):
    """Module title, file name, and the background-task indicator."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(self, text="Dashboard", font=FONT_TITLE, text_color=COLORS["text"], anchor="w")
        self.title_label.grid(row=0, column=0, sticky="w")
        self.file_label = ctk.CTkLabel(self, text="No image loaded", font=FONT_BODY, text_color=COLORS["text"], anchor="e")
        self.file_label.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.copy_label = ctk.CTkLabel(self, text="", font=FONT_SMALL, text_color=COLORS["muted"], anchor="w")
        self.copy_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.status_label = ctk.CTkLabel(self, text="Ready", font=FONT_SMALL, text_color=COLORS["accent"], anchor="e")
        self.status_label.grid(row=1, column=1, sticky="e", padx=(12, 0))
        self.progress = ctk.CTkProgressBar(
            self,
            height=4,
            mode="determinate",
            progress_color=COLORS["accent"],
            fg_color=COLORS["border"],
        )
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.progress.set(0)
        self._running = False

    def render(self, page_id: str, filename: str, status: str, busy: bool) -> None:
        title, copy = PAGE_COPY.get(page_id, ("WaterVision", ""))
        self.title_label.configure(text=title)
        self.copy_label.configure(text=copy)
        self.file_label.configure(text=filename or "No image loaded")
        self.status_label.configure(text=status or "Ready", text_color=COLORS["warning"] if busy else COLORS["accent"])
        if busy and not self._running:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
            self._running = True
        elif not busy and self._running:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(0)
            self._running = False
