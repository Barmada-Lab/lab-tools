from collections import namedtuple

import string
import glob
import csv
import re
import os

import numpy as np

ImgMeta = namedtuple("ImgMeta", ["path", "channel", "time", "col", "row", "montage_idx"])

def get_layout_indexing(scope_name: str, dimension: int) -> np.ndarray:
    """
    constructs an array whose elements index over montage captures for the given microscope and dimension.

    for example, flo uses snake-by-rows, right-and-down indexing for montages.
    So the array indexing a 3x3 mongage on flo is given by:

    1 2 3
    6 5 4
    7 8 9
    """
    default = np.arange(dimension**2).reshape(dimension,dimension) + 1
    snake_by_rows = default.copy()
    snake_by_rows[1::2] = default[1::2,::-1] # reverse every other row

    if scope_name.lower() == 'ixm':
        ''' 1 2 3
            4 5 6
            7 8 9 '''
        return default

    elif scope_name.lower() == 'flo':
        ''' 1 2 3
            6 5 4
            7 8 9 '''
        return snake_by_rows

    elif scope_name.lower() == 'flo2':
        ''' 3 4 9
            2 5 8
            1 6 7 '''
        return snake_by_rows.transpose()[::-1]

    elif scope_name.lower() in {'ds', 'ds2', 'ds3'}:
        ''' 7 8 9
            6 5 4
            1 2 3 '''
        return snake_by_rows[::-1]

    else:
        raise Exception("no matching microscope")


def extract_meta(path) -> ImgMeta:
    """
    Extract image metadata from a standard experiment path
    e.g: experiment_root/raw_imgs/RFP/T1/col_01/A01_01.tif
    """

    pattern = r""" # Verbose regex
        (?P<channel>(RFP)|(GFP))/                   # Pick out the channel
        T(?P<time>\d+)/                             # Pick out the time
        col_(?P<col>\d+)/                           # pick out the row
        (?P<row>[a-z])\d+_(?P<montage_idx>\d{2}).tif  # Pick out the row and montage index from the file name
    """

    search = re.search(pattern, path, re.IGNORECASE | re.VERBOSE)
    if search is None:
        raise Exception("No match found for regex")

    return ImgMeta(
        path,
        search.group("channel"),
        int(search.group("time")),
        int(search.group("col")),
        string.ascii_lowercase.index(search.group("row").lower()),
        int(search.group("montage_idx"))
    )


def find_config_csv(root: str):
    """
    Looks for .csv files in the project root that contain config fields.
    Returns the first.
    """

    for match in glob.glob(os.path.join(root, "*.csv")):
        loaded = load_config_csv(match)
        if 'microscope' in loaded:
            return loaded
    return None

def load_config_csv(path: str):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return next(reader) # Strip off just the first row
