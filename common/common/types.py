from dataclasses import dataclass

from datetime import datetime
from enum import Enum, EnumMeta
import pathlib

@dataclass
class ImgMeta:
    path: pathlib.Path
    channel: str
    time_point: int
    col: int
    row: int
    montage_idx: int

@dataclass
class Exposure:
    channel: str
    exposure_ms: int

@dataclass
class WellSpec:
    label: str
    exposures: list[Exposure]

@dataclass
class MFSpec:
    name: str
    t_transfect: datetime
    objective: str
    microscope: str
    binning: str
    montage_dim: int
    montage_overlap: float
    wells: list[WellSpec]
