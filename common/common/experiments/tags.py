from dataclasses import dataclass
import enum


class Channel(enum.Enum):
    GFP = "GFP"
    RFP = "RFP"
    Cy5 = "Cy5"
    DAPI = "DAPI"
    WHITE = "white_light"

class Objective(enum.Enum):
    X4 = "4X"
    X10 = "10X"
    X20 = "20X"
    X40 = "40X"
    X60 = "60X"

class StackIndexing(enum.Enum):
    TYX = "TYX"
    TZYX = "TZYX"
    TZYXC = "TZYXC"
    TYXC = "TZYXC"
    ZYX = "ZYX"
    ZYXC = "ZYXC"

class Mask(enum.Enum):
    OBJ_PRED = "object-predictions"
    OBJ_TRACK = "object-tracking"


@dataclass(frozen=True)
class Stack:
    indexing: StackIndexing

@dataclass(frozen=True)
class Exposure:
    ms: int

@dataclass(frozen=True)
class PixelBinning:
    n: int

@dataclass(frozen=True)
class Vertex:
    label: str

@dataclass(frozen=True)
class Mosaic:
    """ A stitched mosaic """
    mosaic_dim: int
    overlap: float

@dataclass(frozen=True)
class MosaicTile:
    mosaic_dim: int
    overlap: float
    index: int

@dataclass(frozen=True)
class Timepoint:
    index: int

Tag = Channel | Objective | Stack | Mask | Exposure | PixelBinning | Vertex | MosaicTile | Mosaic | Timepoint
