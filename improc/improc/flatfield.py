from typing import Iterable
import numpy as np
import pathlib
import itertools

from common import WellSpec

from PIL import Image

def background_from_paths(paths: list[pathlib.Path]) -> np.ndarray:
    images = np.array([np.array(Image.open(path)) for path in paths])
    return np.median(images, axis=0, overwrite_input=True).astype(images.dtype)

def trunc_sub(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Truncated integer subtraction.
    Results that would underflow are instead set to zero.
    Supports input dtypes up to uint64
    """
    assert(a1.dtype == a2.dtype)
    assert(a1.shape == a2.shape)

    if a1.dtype == np.uint8:
        projected_dtype = np.int16
    elif a1.dtype == np.uint16:
        projected_dtype = np.int32
    elif a1.dtype == np.uint32:
        projected_dtype = np.int64
    elif a1.dtype == np.uint64:
        projected_dtype = np.int128
    else:
        raise ValueError(f"unsupported datatype {a1.dtype}")

    subtracted = a1.astype(projected_dtype) - a2.astype(projected_dtype)
    subtracted[subtracted < 0] = 0
    return subtracted.astype(a1.dtype)
