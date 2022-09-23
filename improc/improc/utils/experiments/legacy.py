import pathlib
from collections import defaultdict
import pathlib

import string
import re
from typing import Counter
import charset_normalizer
import numpy as np
import tifffile

from .dataset import Dataset
from .experiment import Experiment, ExperimentLoader
from .image import Image, ImgMeta

from ..types import DrugInfo, Exposure, MFSpec, WellSpec
from ..legacy import parse_datetime
from . import tags

"""
spaghetti, spaghetti, spaghetti all around
look away so quickly or your noggin may soon pound
"""


class LegacyImage(Image):

    def __init__(self, path: pathlib.Path, meta: ImgMeta) -> None:
        super().__init__(meta)
        self.path = path

    def load(self) -> np.ndarray:
        return tifffile.imread(self.path)

    def save(self, a: np.ndarray):
        return tifffile.imwrite(self.path, a)


class LegacyLoader(ExperimentLoader):

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self.path = path
        self.name = path.name

    def _try_load_mfile(self) -> MFSpec:
        experiment_name = self.name
        glob = f"*{experiment_name}.csv"
        try:
            csv = next(self.path.glob(glob))
        except StopIteration:
            raise Exception(f'No csv of the form "{glob}" found in {self.path}')
        return read_mfile(csv)

    def _load_datasets(self, mfile: MFSpec) -> list[Image]:
        raw_imgs: list[Image] = [LegacyImage(path, extract_meta_rawpath(mfile, path)) for path in self.path.glob("raw_imgs/**/*.tif")]
        processed_imgs: list[Image] = []
        return raw_imgs

    def load(self) -> Experiment:
        mfile = self._try_load_mfile()
        datasets = self._load_datasets(mfile)
        return Experiment(
            self.name,
            datasets,
            mfile
        )


def extract_meta_rawpath(mfile: MFSpec, path: pathlib.Path) -> ImgMeta :
    """
    Extract image metadata from a standard experiment path
    e.g: experiment_root/raw_imgs/RFP/T1/col_01/A01_01.tif
    """

    pattern = r"""# Verbose regex
        (?P<channel>(GFP)|(RFP)|(Cy5)|(white_light)|(DAPI))/ # Pick out the channel
        T(?P<timepoint>\d+)/                                 # Pick out the time
        col_(?P<col>\d+)/                                    # pick out the row
        (?P<row>[a-z])\d+_(?P<montage_idx>\d{2}).tif         # Pick out the row and montage index from the file name
    """

    search = re.search(pattern, path.__str__(), re.IGNORECASE | re.VERBOSE)
    if search is None:
        raise Exception(f"Failed to parse path {path}")

    montage_idx = int(search.group("montage_idx"))
    row, col = search.group("row"), search.group("col")
    label = row + col
    timepoint = int(search.group("timepoint"))
    channel = search.group("channel")
    exposures = next(filter(lambda x: x.label == label, mfile.wells)).exposures
    exposure = next(filter(lambda x: x.channel == channel, exposures)).exposure_ms

    return ImgMeta([
        tags.Channel(channel),
        tags.Timepoint(timepoint - 1),
        tags.Vertex(label),
        tags.Exposure(exposure),
        tags.MosaicTile(mfile.montage_dim, mfile.montage_overlap, montage_idx - 1)
    ])


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
        return None

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

        drugs = []
        for idx in range(2):
            drug = well_spec[f"Drug{idx+1}"]
            if drug.lower() in ["na","n/a","0"]:
                continue
            drug_conc = well_spec[f"[Drug{idx+1}]"]
            if drug_conc.lower() in ["na", "n/a", "0"]:
                drug_conc = None
            else:
                drug_conc = float(drug_conc)
            drugs.append(DrugInfo(drug, drug_conc))

        return WellSpec(label, wells, drugs)

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
