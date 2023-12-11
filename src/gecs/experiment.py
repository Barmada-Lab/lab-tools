import enum

class Axes(str, enum.Enum):
    X = "x"
    Y = "y"
    Z = "z"
    FIELD = "field"
    REGION = "region"
    CHANNEL = "channel"
    TIME = "time"
    RGB = "rgb"

class ExperimentType(enum.Enum):
    LEGACY = "legacy"
    LEGACY_ICC = "legacy-icc"
    ND2 = "nd2"
    LUX = "lux"
    CQ1 = "cq1"
