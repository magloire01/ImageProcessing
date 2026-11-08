"""Synthetic scene used to exercise the three segmentation methods.

The picture is generated, not photographed, so the application can be
demonstrated and tested without an external file. Geometry is deterministic
for a given seed.

The scene is constructed to make a specific point:

- The lake and the river are smooth and blue-green. Color and frequency agree.
- The sandbar is smooth and tan. Frequency accepts it; hue rejects it.
- The land is textured and yellow-green. Frequency rejects it. A wide blue
  hue interval would also reject most of it, because its hue sits below the
  primary water range.

A ground-truth mask of the lake, river, and a small dark pond is returned
with the image. The sandbar is intentionally absent from that mask.
"""

from __future__ import annotations

import numpy as np


def _ellipse(height: int, width: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0


def demo_regions(shape: tuple[int, int] = (480, 640)) -> dict[str, np.ndarray]:
    """Boolean regions of the synthetic scene, independent of pixel noise.

    Keys: lake, river, pond, sand, water. `water` is the ground truth
    (lake, river, and the dark pond). `sand` is smooth and dry on purpose.
    """
    height, width = shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    lake = _ellipse(height, width, cy=250, cx=380, ry=110, rx=170)
    river = (np.abs(y * height - (300.0 + 0.15 * (x * width - 380.0))) <= 11.0) & (x * width >= 470)
    pond = _ellipse(height, width, cy=400, cx=90, ry=28, rx=36)
    sand = _ellipse(height, width, cy=78, cx=120, ry=48, rx=90)
    return {
        "lake": lake,
        "river": river,
        "pond": pond,
        "sand": sand,
        "water": lake | river | pond,
    }


def make_demo_scene(
    shape: tuple[int, int] = (480, 640),
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8, ground-truth water mask).

    The ground truth is the geometric water that was painted (lake, river,
    and dark pond). It is a reference annotation, not a detector output.
    """
    height, width = shape
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Textured land. Hue is kept near yellow-green, outside the default
    # primary water interval (90°–255°) and outside the narrow ochre band.
    land = np.zeros((height, width, 3), dtype=np.float32)
    land[..., 0] = 150.0
    land[..., 1] = 132.0
    land[..., 2] = 48.0
    texture = rng.normal(0.0, 22.0, size=(height, width, 1)).astype(np.float32)
    # A second, finer noise layer so the local spectrum is genuinely broad.
    fine = rng.normal(0.0, 10.0, size=(height, width, 1)).astype(np.float32)
    land = land + texture + fine

    # Slow spatial shade, almost no high frequency, tan color.
    sand = np.zeros((height, width, 3), dtype=np.float32)
    sand[..., 0] = 214.0 + 8.0 * x
    sand[..., 1] = 196.0 + 6.0 * y
    sand[..., 2] = 150.0
    sand += rng.normal(0.0, 1.2, size=sand.shape).astype(np.float32)
    regions = demo_regions(shape)
    sand_region = regions["sand"]

    # Principal lake: blue-green gradient (deeper toward the center) plus a
    # negligible noise floor. Hue stays near 200°.
    water = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.clip(1.0 - np.sqrt((y - 0.52) ** 2 + (x - 0.58) ** 2) * 1.6, 0.35, 1.0)
    water[..., 0] = 28.0 + 10.0 * (1.0 - depth)
    water[..., 1] = 86.0 + 40.0 * depth
    water[..., 2] = 128.0 + 50.0 * depth
    water += rng.normal(0.0, 1.0, size=water.shape).astype(np.float32)
    lake = regions["lake"]
    river = regions["river"]

    # Nearly grey. Saturation is below the primary gate, so only the
    # dark-water rule has a reason to accept these pixels. The surface is smooth.
    dark = np.zeros((height, width, 3), dtype=np.float32)
    dark[..., 0] = 40.0
    dark[..., 1] = 41.0
    dark[..., 2] = 42.0
    dark += rng.normal(0.0, 0.4, size=dark.shape).astype(np.float32)
    pond = regions["pond"]

    scene = land.copy()
    scene[sand_region] = sand[sand_region]
    # Paint river and lake with the chromatic water color, pond with grey.
    # The ground truth is this geometric set, not a threshold of the detector.
    chromatic = lake | river
    scene[chromatic] = water[chromatic]
    scene[pond] = dark[pond]

    rgb = np.clip(np.round(scene), 0, 255).astype(np.uint8)
    truth = chromatic | pond
    return rgb, truth
