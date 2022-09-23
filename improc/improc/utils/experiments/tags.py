from dataclasses import dataclass
from datetime import datetime
import enum

from tifffile.tifffile import datetime

class Channel(str, enum.Enum):
    GFP = "GFP"
    RFP = "RFP"
    Cy5 = "Cy5"
    DAPI = "DAPI"
    WHITE = "white_light"

class Objective(str, enum.Enum):
    X4 = "4X"
    X10 = "10X"
    X20 = "20X"
    X40 = "40X"
    X60 = "60X"

class StackIndex(str, enum.Enum):
    T = "T"
    XY = "XY"
    Z = "Z"
    C = "C"
    M = "M"

@dataclass(frozen=True, eq=True)
class Exposure:
    ms: int

@dataclass(frozen=True, eq=True)
class StackMeta:
    indexing: list[StackIndex]

@dataclass(frozen=True, eq=True)
class ChannelMeta:
    indexing: list[Channel]
    exposures: list[Exposure]

@dataclass(frozen=True, eq=True)
class TimeseriesMeta:
    indexing: list[datetime.datetime]

@dataclass(frozen=True, eq=True)
class ZSliceMeta:
    offsets: list[float]

@dataclass(frozen=True, eq=True)
class PlanarMeta:
    pixel_binning: int
    pixel_size_um: float

@dataclass(frozen=True, eq=True)
class MosaicMeta:
    indexing: list[tuple[int,int]]
    overlap: float

@dataclass(frozen=True, eq=True)
class Vertex:
    label: str

@dataclass(frozen=True, eq=True)
class Mosaic:
    mosaic_dim: int
    overlap: float

@dataclass(frozen=True, eq=True)
class MosaicTile:
    mosaic_dim: int
    overlap: float
    index: int

@dataclass(frozen=True, eq=True)
class Timepoint:
    index: int

Tag = Mosaic | Channel | Objective | StackMeta | Exposure | Vertex | MosaicTile | ZSliceMeta | Timepoint
