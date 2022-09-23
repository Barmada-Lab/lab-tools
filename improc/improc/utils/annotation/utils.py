from typing import Tuple

import numpy as np
import pathlib
import math

from typing import Union
from cv2 import cv2
from . import constants
from .types import Point, Roi


def determine_file_encoding(filename: str) -> Union[str, None]:
    if filename.endswith(".pkl") or filename.endswith(".p"):
        return constants.PKL_ENC
    elif filename.endswith(".json"):
        return constants.JSON_ENC
    elif filename.endswith(".roi"):
        return constants.IJ_ENC
    else:
        return None

def extract_well_and_id(path: pathlib.Path) -> Tuple[str, int]:
    well, id = path.name.split('.')[0].split('-')
    return well, int(id)

def make_rois(data):
    #+TAG:DIRTY
    for d in data:
        # Extract radius and center coordinates.
        r, x, y = d['r'], int(d['x']), int(d['y'])
        r = math.floor(r)
        # Draw a circle on a mask and find its contour to determine contour in correct format.
        mask = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] # There's only one contour :)
        # Add the center coordinates and remove the radius. The circle was drawn with a center
        # at the radius so a full circle could be drawn, so a full contour could be extracted.
        contour[:, :, 0] += x - r
        contour[:, :, 1] += y - r
        center = Point(x,y)
        yield Roi(center, contour[:,0,:])

def confirm_overwrite(path: pathlib.Path) -> bool:
    choice = None
    while choice is None:
        _input = input(f"{path} already exists; are you sure you want to overwrite it? [y/N]")
        if _input.lower().strip() in ["y", "ye", "yes"]:
            choice = True
        elif _input.lower().strip() in ["n", "no"]:
            choice = False
    return choice
