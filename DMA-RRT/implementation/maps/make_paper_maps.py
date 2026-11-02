"""
    Make the maps of Fig. 4 and Fig. 7(a) of the paper, read from the figures

    The paper gives no dimensions. A robot circle is about 6 pixels in the figures, so we use 6 pixels = 1 m for
    our robots of radius 1 m. All positions are pixels of the paper's page at the resolution we read them.

    Run from DMA-RRT/implementation:
    python3 maps/make_paper_maps.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

RESOLUTION = 0.1  # [m/cell]
METERS_PER_PIXEL = 1.0 / 6.0
FREE, OCCUPIED = 254, 0

Pixel = Tuple[float, float]

# frame of the figure panel: left, top, right, bottom
HALLWAY_FRAME = (252.0, 93.0, 465.0, 211.0)
HALLWAY_OBSTACLES: List[List[Pixel]] = [
    [(326, 93), (403, 93), (403, 118), (326, 118)],
    [(326, 142), (403, 142), (403, 211), (326, 211)],
]
OBSTACLES_FRAME = (258.0, 75.0, 399.0, 403.0)
OBSTACLES_OBSTACLES: List[List[Pixel]] = [
    [(268, 158), (300, 158), (300, 189), (268, 189)],
    [(334, 135), (379, 160), (362, 190), (322, 165)],
    [(340, 222), (367, 247), (339, 276), (311, 250)],
    [(366, 268), (399, 268), (399, 299), (366, 299)],
    [(268, 287), (284, 287), (284, 318), (268, 318)],
    [(328, 316), (355, 342), (327, 371), (301, 345)],
]


def write_map(name: str, frame: Tuple[float, float, float, float], obstacles: List[List[Pixel]]) -> None:
    left, top, right, bottom = frame
    cells_per_pixel = METERS_PER_PIXEL / RESOLUTION
    width = int(round((right - left) * cells_per_pixel))
    height = int(round((bottom - top) * cells_per_pixel))
    image = Image.new("L", (width, height), FREE)
    draw = ImageDraw.Draw(image)
    for polygon in obstacles:
        draw.polygon([((x - left) * cells_per_pixel, (y - top) * cells_per_pixel) for x, y in polygon], fill=OCCUPIED)

    directory = Path(__file__).resolve().parent / name
    directory.mkdir(exist_ok=True)
    Image.fromarray(np.asarray(image)).save(directory / f"{name}.pgm")
    (directory / "map.yaml").write_text(
        f"image: {name}.pgm\nresolution: {RESOLUTION}\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )


if __name__ == "__main__":
    write_map("hallway", HALLWAY_FRAME, HALLWAY_OBSTACLES)
    write_map("four_agents", OBSTACLES_FRAME, OBSTACLES_OBSTACLES)
