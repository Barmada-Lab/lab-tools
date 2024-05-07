from cytomancer.enumero import NaturalOrderStrEnum


class Axes(NaturalOrderStrEnum):
    REGION = "region"
    FIELD = "field"
    CHANNEL = "channel"
    TIME = "time"
    Y = "y"
    X = "x"
    Z = "z"
    RGB = "rgb"


class ExperimentType(NaturalOrderStrEnum):
    CQ1 = "cq1"
    ND2 = "nd2"
    LUX = "lux"
    LEGACY = "legacy"
    LEGACY_ICC = "legacy-icc"
