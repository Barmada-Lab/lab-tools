from dataclasses import dataclass

from datetime import datetime
from enum import Enum
import pathlib

class Channel(str, Enum):
    GFP = "GFP"
    RFP = "RFP"
    Cy5 = "Cy5"
    DAPI = "DAPI"
    WhiteLight = "white_light"

@dataclass
class ImgMeta:
    channel: str
    time: datetime
    col: int
    row: int
    montage_idx: int

@dataclass
class Exposure:
    channel: Channel
    exposure_ms: int

@dataclass
class WellSpec:
    label: str
    exposures: list[Exposure]

@dataclass
class MFSpec:
    name: str
    t_transfect: datetime
    microscope: str
    binning: str
    montage_dim: int
    montage_overlap: float
    wells: list[WellSpec]
