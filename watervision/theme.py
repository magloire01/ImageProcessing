"""Visual constants for the WaterVision desktop interface.

The palette is a dark scientific dashboard: deep navy surfaces with blue
and cyan accents. Widgets read these values so spacing, type, and color
stay consistent across modules.
"""

from __future__ import annotations

COLORS = {
    "bg": "#0B1220",
    "sidebar": "#0C1424",
    "card": "#121C2E",
    "stage": "#0A101C",
    "border": "#1E3A5F",
    "accent": "#22D3EE",
    "accent_dim": "#0E7490",
    "blue": "#3B82F6",
    "blue_hover": "#2563EB",
    "text": "#E8F1FF",
    "muted": "#8BA3C7",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "violet": "#A78BFA",
    "amber": "#F59E0B",
}

FONT_FAMILY = "Segoe UI"

FONT_LOGO = (FONT_FAMILY, 18, "bold")
FONT_TITLE = (FONT_FAMILY, 22, "bold")
FONT_SECTION = (FONT_FAMILY, 15, "bold")
FONT_CARD = (FONT_FAMILY, 13, "bold")
FONT_BODY = (FONT_FAMILY, 13)
FONT_SMALL = (FONT_FAMILY, 11)
FONT_KPI = (FONT_FAMILY, 26, "bold")
FONT_BUTTON = (FONT_FAMILY, 13, "bold")

# Cyan used when a mask is painted onto a dark background or over a photo.
WATER_RGB = (34, 211, 238)
MASK_BG_RGB = (12, 20, 32)
