from pathlib import Path
import os

import click
import numpy as np
from skimage import exposure # type: ignore
import tifffile

from ..core.helpers import img_path_loader

def project_sum(stack: np.ndarray):
    arr = stack.copy().astype(np.float32)
    return np.sum(arr, axis=0)

def project_mip(stack: np.ndarray):
    return np.max(stack, axis=0)

@click.command("project")
@click.argument("stacks", nargs=-1, type=click.Path(path_type=Path, exists=True))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path.cwd())
@click.option("--mode", "-m", type=click.Choice(["max", "sum"]), default="max")
def cli_entry(stacks: list[Path], output_dir: Path, mode: str):
    os.makedirs(output_dir, exist_ok=True)
    for path in img_path_loader(stacks):
        img = tifffile.imread(path)
        projected = project_mip(img) if mode == "max" else project_sum(img)
        tifffile.imsave(output_dir/ path.name, projected)