"""Image acquisition and raster export.

Pillow reads and writes the files. NumPy owns the arrays the moment a file
is in memory. Multi-page TIFF files contribute their first frame only; the
caller is told so the limitation is visible rather than silent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ImageIOError(Exception):
    """A file that cannot be used as an analysis image."""


def load_rgb(path: str) -> tuple[np.ndarray, str]:
    """Load a JPG, PNG, TIFF, or BMP file as uint8 RGB.

    EXIF orientation is applied first, so a phone photograph is not analyzed
    on its side. Palette, grayscale, and 16-bit files are reduced to 8-bit
    RGB by Pillow's converter, which scales 16-bit samples into 0–255.
    An alpha channel is composited over black: transparent pixels carry no
    water observation and must not become an invented color.

    Returns the array and a short note (empty when there is nothing to add).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageIOError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageIOError(
            "Unsupported file type. Load a JPG, JPEG, PNG, TIFF, or BMP image."
        )
    try:
        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image)
            note = ""
            n_frames = int(getattr(image, "n_frames", 1) or 1)
            if n_frames > 1:
                image.seek(0)
                note = "Multi-page file: only the first frame was loaded."
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (0, 0, 0))
                background.paste(image, mask=image.getchannel("A"))
                rgb_image = background
            else:
                rgb_image = image.convert("RGB")
            array = np.asarray(rgb_image, dtype=np.uint8)
    except ImageIOError:
        raise
    except Exception as exc:  # Pillow raises several private error types.
        raise ImageIOError(f"This file could not be read as an image. {exc}") from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageIOError("The file did not produce a 3-channel RGB image.")
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise ImageIOError("The image is smaller than 8×8 pixels and cannot be analyzed.")
    return np.ascontiguousarray(array), note


def load_mask(path: str) -> np.ndarray:
    """Load a reference mask. Non-zero (or above mid-grey) pixels are water.

    The result is Boolean at the file's native resolution. The evaluation
    page resizes it with nearest-neighbor if the working image differs, and
    it says so when that happens.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageIOError(f"File not found: {file_path}")
    try:
        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ("1", "L"):
                array = np.asarray(image)
            else:
                array = np.asarray(image.convert("L"))
    except Exception as exc:
        raise ImageIOError(f"The reference mask could not be read. {exc}") from exc
    if array.ndim != 2:
        raise ImageIOError("The reference mask must be a single-channel image.")
    if array.dtype == np.bool_:
        return array
    # Mode "1" may already be bool. Integer masks use a midpoint threshold so
    # both 0/1 and 0/255 annotations work.
    if np.issubdtype(array.dtype, np.integer):
        return array > 127
    return array.astype(np.float32) > 0.5


def save_rgb(path: str, rgb: np.ndarray) -> None:
    """Write an RGB or single-channel uint8 image. PNG is the safe default."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, mode="RGB")
    else:
        raise ImageIOError("Only single-channel or RGB images can be saved.")
    try:
        image.save(path)
    except Exception as exc:
        raise ImageIOError(f"The image could not be written. {exc}") from exc


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Paint a binary mask: cyan water on a dark field, for preview and export."""
    binary = np.asarray(mask, dtype=bool)
    canvas = np.empty((*binary.shape, 3), dtype=np.uint8)
    canvas[..., 0] = 12
    canvas[..., 1] = 20
    canvas[..., 2] = 32
    canvas[binary] = (34, 211, 238)
    return canvas


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Keep the water, paint everything else black, and draw the contour.

    Non-water pixels are set to 0. Water pixels keep the original color, so
    the surface itself stays visible. The contour is the morphological
    gradient of the mask, a band two pixels thick lying inside the water:

        edge = A \\ (A ⊖ B ⊖ B)

    That band is drawn white, on the water, not on the black background.
    `alpha` is unused; it remains so older call sites keep the same signature.
    """
    from scipy.ndimage import binary_erosion

    del alpha  # the result is a hard separation, not a translucent tint
    base = np.asarray(rgb)
    if base.ndim != 3:
        raise ImageIOError("The overlay expects an RGB image.")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != base.shape[:2]:
        raise ImageIOError("The mask and the image have different sizes.")
    out = np.zeros_like(base)
    out[binary] = base[binary]
    eroded = binary_erosion(binary, iterations=2, border_value=0)
    edge = binary & ~eroded
    out[edge] = (255, 255, 255)
    return out


def mask_to_label_image(mask: np.ndarray) -> np.ndarray:
    """uint8 label image, 255 = water, 0 = background, suitable for a dataset."""
    return (np.asarray(mask, dtype=bool).astype(np.uint8)) * np.uint8(255)


def image_info(rgb: np.ndarray, path: str = "") -> dict:
    """Dimensions, channel count, and file size for the acquisition panel."""
    height, width = rgb.shape[:2]
    info = {
        "width": int(width),
        "height": int(height),
        "channels": 3,
        "dtype": "8-bit RGB",
        "megapixels": (width * height) / 1_000_000.0,
    }
    if path:
        file_path = Path(path)
        if file_path.is_file():
            info["bytes"] = file_path.stat().st_size
        else:
            info["bytes"] = None
    else:
        info["bytes"] = None
    return info
