from dataclasses import dataclass

import numpy as np

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Roi:
    centroid: Point
    contour: np.ndarray

@dataclass
class Roi_ts:
    """Series of rois indexed by time"""
    id: str
    rois: list[Roi]
