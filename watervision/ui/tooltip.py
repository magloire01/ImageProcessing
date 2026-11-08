"""Small hover labels for controls that encode a formula or a unit.

CustomTkinter has no tooltip widget. This is a borderless window shown
under the control and destroyed when the pointer leaves. It never takes
focus, so it does not interrupt a slider drag.
"""

from __future__ import annotations

import tkinter as tk

from watervision.theme import COLORS, FONT_SMALL


class Tooltip:
    """Attach explanatory text to any Tk widget."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self._tip: tk.Toplevel | None = None
        if text:
            widget.bind("<Enter>", self._show, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(bg=COLORS["border"])
        label = tk.Label(
            tip,
            text=self.text,
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=FONT_SMALL,
            justify="left",
            wraplength=340,
            padx=10,
            pady=6,
        )
        label.pack(padx=1, pady=1)
        self._tip = tip

    def _hide(self, _event=None) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None
