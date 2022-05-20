from collections import defaultdict
from datetime import datetime
import pathlib

import string
import re
from typing import Counter

from common.types import Exposure, MFSpec, WellSpec, ImgMeta
from common.utils.legacy import parse_datetime

import charset_normalizer

def try_load_mfile(path: pathlib.Path) -> MFSpec:
    experiment_name = path.name
    glob = f"*{experiment_name}.csv"
    try:
        csv = next(path.glob(glob))
    except StopIteration:
        raise Exception(f'No csv of the form "{glob}" found in {path}')
    return read_mfile(csv)

def extract_meta(path: pathlib.Path) -> ImgMeta:
    """
    Extract image metadata from a standard experiment path
    e.g: experiment_root/raw_imgs/RFP/T1/col_01/A01_01.tif
    """

    pattern = r"""# Verbose regex
        (?P<channel>(GFP)|(RFP)|(Cy5)|(white_light)|(DAPI))/ # Pick out the channel
        T(?P<time>\d+)/                                      # Pick out the time
        col_(?P<col>\d+)/                                    # pick out the row
        (?P<row>[a-z])\d+_(?P<montage_idx>\d{2}).tif         # Pick out the row and montage index from the file name
    """

    search = re.search(pattern, path.__str__(), re.IGNORECASE | re.VERBOSE)
    if search is None:
        raise Exception("No match found for regex")

    return ImgMeta(
        path = path,
        channel = search.group("channel"),
        time_point = int(search.group("time")),
        col = int(search.group("col")),
        row = string.ascii_lowercase.index(search.group("row").lower()),
        montage_idx = int(search.group("montage_idx"))
    )

def read_mfile(path: pathlib.Path) -> MFSpec:
    # There doesn't seem to be a good way to automatically detect character encoding
    # using vanilla python... so we use this
    lines = str(charset_normalizer.from_path(path).best()).split("\n")

    def tokenize(line: str) -> list[str]:
        return [token.strip() for token in line.split(",")]

    def deduplicate(arr) -> list[str]:
        seen = defaultdict(lambda: 0)
        counter = Counter(arr)
        dedup = []
        for x in arr:
            if counter[x] == 1:
                dedup.append(x)
            else:
                count = seen[x]
                dedup.append(f"{x}-{count}")
                seen[x] += 1
        return dedup

    def assoc(fields_str: str, attrs_str: str) -> dict[str, str]:
        fields = deduplicate(tokenize(fields_str))
        attrs = tokenize(attrs_str)
        return dict(zip(fields, attrs))

    gen_spec = assoc(lines[0], lines[1])

    def valid_line(line):
        return not all([token.strip() == "" for token in line.split(",")])

    if not valid_line(lines[3]):
        raise Exception("malformed csv; fourth line should contain keys for well configs")

    well_specs = [assoc(lines[3], line) for line in lines[4:] if valid_line(line)]

    name = gen_spec["PlateID"]
    t_transfect = parse_datetime(gen_spec["Transfection date"], gen_spec["Transfection time"])
    objective = gen_spec["Objective"]
    microscope = gen_spec["microscope"]
    binning = gen_spec["binning"]
    montage_dim = int(gen_spec["Montage XY"])
    montage_overlap = 1.0 / int(gen_spec["Tile overlap"])

    def build_wellspec(well_spec: dict[str,str]) -> WellSpec:
        label = well_spec["Well"]
        wells = []
        for idx in range(4):
            fp = well_spec[f"FP{idx+1}"]
            if fp.lower() not in {"cy5", "gfp", "rfp", "dapi", "white_light"}:
                continue
            try:
                exposure = int(well_spec[f"Exposure (ms)-{idx}"])
            except:
                continue
            wells.append(Exposure(fp, exposure))
        return WellSpec(label, wells)

    return MFSpec(
        name=name,
        t_transfect = t_transfect,
        objective = objective,
        microscope = microscope,
        binning = binning,
        montage_dim = montage_dim,
        montage_overlap = montage_overlap,
        wells = [build_wellspec(well_spec) for well_spec in well_specs]
    )
