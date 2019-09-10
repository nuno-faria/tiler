from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import math
import conf
import multiprocessing

import click


# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = conf.DEPTH
# list of rotations, in degrees, to apply over the original image
ROTATIONS = conf.ROTATIONS
THREADS = multiprocessing.cpu_count()


def get_tile_dir(img_dir: Path, img_name: str) -> Path:
    tile_dir = img_dir / Path(f"gen_{img_name}")

    if not tile_dir.exists():
        tile_dir.mkdir()

    return tile_dir


def get_img(img: Path) -> np.array:
    img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    return img.astype("float")


def get_dimensions(img: np.array) -> Tuple[int, int, Tuple[float, float]]:
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    return height, width, center


def make_rotation(
    img_name: str,
    ext: str,
    tile_dir: Path,
    new_img: np.array,
    dimensions: Tuple[int, int, Tuple[float, float]],
    rotation: float,
    colors: Tuple[float, float, float],
    bar: tqdm,
):
    height, width, center = dimensions
    b, g, r = colors

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]
    cv2.imwrite(
        f"{tile_dir}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}",
        cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h)),
        # compress image
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    bar.update()


def generate_tiles(
    img: np.array,
    img_name: str,
    ext: str,
    tile_dir: Path,
    depth: int,
    rotations: List[int],
    pool: ThreadPoolExecutor,
):
    dimensions = get_dimensions(img)
    b_range = np.arange(0, 1.01, 1 / depth)
    g_range = np.arange(0, 1.01, 1 / depth)
    r_range = np.arange(0, 1.01, 1 / depth)
    operations = len(b_range) ** 3 * len(rotations)
    progress_bar = tqdm(total=operations)

    for b, g, r in product(b_range, g_range, r_range):
        colors = b, g, r
        new_img = img * [b, g, r, 1]
        new_img = new_img.astype("uint8")
        for rotation in rotations:
            pool.submit(
                make_rotation,
                img_name,
                ext,
                tile_dir,
                new_img,
                dimensions,
                rotation,
                colors,
                progress_bar,
            )


@click.command()
@click.option(
    "-d",
    "--depth",
    default=DEPTH,
    help="Color depth.",
    show_default=True,
    type=click.INT,
)
@click.option(
    "-r",
    "--rotations",
    default=ROTATIONS,
    help="Rotations.",
    multiple=True,
    show_default=True,
    type=click.INT,
)
@click.argument("img", type=click.Path(exists=True))
def cmd(img: str, depth: int, rotations: List[int]):
    img_path = Path(img)
    img_dir = img_path.parent
    img_name, ext = img_path.name.split(".")
    tile_dir = get_tile_dir(img_dir, img_name)
    img = get_img(img_path)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        generate_tiles(img, img_name, ext, tile_dir, depth, rotations, pool)


if __name__ == "__main__":
    cmd()
