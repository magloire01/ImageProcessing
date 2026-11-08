"""Binary morphology on a water mask.

The structuring element B is a disk built with NumPy. Erosion and dilation
themselves are the SciPy binary filters, which implement the set definitions
below. Opening, closing, hole filling, and small-component removal are
composed in this module so the order is visible and can be justified.

Notation: A is the set of pixels currently labeled water. B is the disk.
A_z is A translated so that its origin sits at pixel z.

    dilation   A ⊕ B = { z | A ∩ B_z ≠ ∅ }     grows A by the radius of B
    erosion    A ⊖ B = { z | B_z ⊆ A }          shrinks A by the radius of B
    opening    A ∘ B = (A ⊖ B) ⊕ B              deletes objects thinner than B
    closing    A • B = (A ⊕ B) ⊖ B              fills gaps thinner than B

Opening is applied before closing. Specular sparkles are small false
positives and should be removed before closing has a chance to weld them
into a lake. Closing then bridges breaks caused by glare and thin foam.
Holes that are fully enclosed (a bright reflection inside a lake) are filled
next. Components smaller than a pixel count are removed last, after the
shape operations have had their effect, so the count refers to objects the
user will actually see.

Connectivity for component labeling is 8: a pixel touches its diagonal
neighbors. A diagonal string of water pixels is one region, which matches
the way a river can cross a pixel grid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def disk(radius: int) -> np.ndarray:
    """Boolean disk of the given radius, used as the structuring element.

    The center pixel is the origin. A pixel at offset (y, x) belongs to the
    disk when y² + x² ≤ r². Radius 0 yields a single pixel, and every
    morphological operation with that element is the identity.
    """
    radius = int(max(0, radius))
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    offset = np.arange(-radius, radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(offset, offset, indexing="ij")
    return (yy * yy + xx * xx) <= radius * radius


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erosion A ⊖ B. The border outside the image is treated as background,
    so objects shrink and objects touching the border are not artificially
    preserved.
    """
    structure = disk(radius)
    return ndimage.binary_erosion(np.asarray(mask, dtype=bool), structure=structure, border_value=0)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation A ⊕ B, with the outside of the image treated as background."""
    structure = disk(radius)
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=structure, border_value=0)


def opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Opening = dilation(erosion(A)). Removes positive noise thinner than B."""
    return dilate(erode(mask, radius), radius)


def closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Closing = erosion(dilation(A)). Fills negative gaps thinner than B."""
    return erode(dilate(mask, radius), radius)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background pockets that cannot reach the image border.

    SciPy floods the complement from the border. Any complement pixel the
    flood does not reach is enclosed by water and is added to the mask.
    A bay that opens onto the border is not a hole and is left alone. That
    distinction matters for a river mouth or a lake cut by the photo edge.
    """
    return ndimage.binary_fill_holes(np.asarray(mask, dtype=bool))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels. Background is 0. Labels are 1 … N."""
    structure = np.ones((3, 3), dtype=int)
    labeled, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labeled, int(count)


def remove_small_regions(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected components whose area is strictly below min_pixels.

    Area here is the cardinality of the component, in pixels. The threshold
    is a cleaning parameter, not a physical area. Physical area is computed
    later, and only when a pixel size in meters is supplied.
    """
    min_pixels = int(max(0, min_pixels))
    binary = np.asarray(mask, dtype=bool)
    if min_pixels <= 1:
        return binary
    labeled, count = label_components(binary)
    if count == 0:
        return binary
    areas = np.bincount(labeled.ravel())
    keep = areas >= min_pixels
    keep[0] = False  # label 0 is the background, never water
    return keep[labeled]


def component_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (number of regions, size of the largest region) in pixels.

    An empty mask has 0 regions and a largest size of 0.
    """
    labeled, count = label_components(mask)
    if count == 0:
        return 0, 0
    areas = np.bincount(labeled.ravel())
    # areas[0] is the background count and is excluded.
    return count, int(areas[1:].max())


def apply_morphology(mask: np.ndarray, params: dict) -> np.ndarray:
    """Apply the user-selected sequence. Disabled steps are skipped.

    The order is fixed: opening, closing, hole filling, small-region removal.
    The radius is shared by opening and closing so the same physical scale
    (in pixels) is used to delete noise and to bridge gaps.
    """
    result = np.asarray(mask, dtype=bool)
    radius = int(params.get("morph_radius", 0))
    if radius > 0 and bool(params.get("do_opening", True)):
        result = opening(result, radius)
    if radius > 0 and bool(params.get("do_closing", True)):
        result = closing(result, radius)
    if bool(params.get("do_fill_holes", True)):
        result = fill_holes(result)
    if bool(params.get("do_remove_small", True)):
        result = remove_small_regions(result, int(params.get("min_region_pixels", 0)))
    return result
