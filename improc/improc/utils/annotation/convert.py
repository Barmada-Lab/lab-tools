from collections import defaultdict
from typing import Tuple
from zipfile import ZipFile

import sys
import json
import struct
import shutil
import pathlib
import tempfile

from .ij_coding import encode as encode_ij
from .types import Roi_ts, Roi
from . import utils

def read_json(src: pathlib.Path) -> Roi_ts:
    well, id = utils.extract_well_and_id(src)
    with open(src) as f:
        rois = json.load(f)
    return Roi_ts(f"{well}_{id}", list(utils.make_rois(rois["data"])))

def read_pkl(src: pathlib.Path):
    pass

def read_ij(src: pathlib.Path):
    pass

def write_ij(roi_ts: Roi_ts, dest: pathlib.Path):
    archive = ZipFile(dest, mode="a")
    for idx, roi in enumerate(roi_ts.rois):
        encoded = encode_ij(roi, roi_ts.id)
        out_path = f"T{idx+1}_{roi_ts.id}.roi"
        with archive.open(out_path, "w") as f:
            f.write(encoded)

def json_to_ij(src: pathlib.Path, dest: pathlib.Path):
    print(f"converting {src}")
    rois = read_json(src)
    write_ij(rois, dest)
