from improc.enumero import NaturalOrderStrEnum
from toolz import curry

import xarray as xr
import numpy as np


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
    LEGACY = "legacy"
    LEGACY_ICC = "legacy-icc"
    ND2 = "nd2"
    LUX = "lux"
    CQ1 = "cq1"


# ex. field-1|region-B02|channel-GFP:RFP:Cy5|time-1:2:3:4:5:6:7:8:9:10
FIELD_DELIM = "|"
FIELD_VALUE_DELIM = "-"
VALUE_DELIM = ":"


def _fmt_coord_selector_str(label, coord_arr):
    arr = np.atleast_1d(coord_arr)
    if label == Axes.TIME:
        arr = arr.astype("long")
    if np.issubdtype(arr.dtype, np.str_):
        for value in arr:
            assert FIELD_DELIM not in value, f"{label} value {value} is invalid; contains a '|'; rename and try again"
            assert FIELD_VALUE_DELIM not in value, f"{label} value {value} is invalid; contains a '-'; rename and try again"
            assert VALUE_DELIM not in value, f"{label} value {value} is invalid; contains a ':'; rename and try again"

    return f"{label}{FIELD_VALUE_DELIM}" + VALUE_DELIM.join(map(str, arr))


@curry
def _parse_field_selector(arr: xr.DataArray | None, selector: str):
    field_name, field_values = selector.split(FIELD_VALUE_DELIM)
    try:
        axis = Axes(field_name)
    except ValueError:
        raise ValueError(f"Invalid field name {field_name} in selector {selector}")

    if arr is None:
        target_dtype = np.str_
    else:
        target_dtype = arr[axis].dtype

    field_value_tokens = np.array(field_values.split(VALUE_DELIM)).astype(target_dtype)
    if field_value_tokens.size == 1:
        field_value = field_value_tokens[0]
        return (axis, field_value)
    else:
        return (axis, field_value_tokens)


def coord_selector(arr: xr.DataArray) -> str:
    """Derives a string-formatted selector from an array's coordinates."""
    coords = sorted(arr.coords.items())
    filtered = filter(lambda coord: coord[0] not in [Axes.X, Axes.Y], coords)
    return FIELD_DELIM.join([
        _fmt_coord_selector_str(axis, coord.values) for axis, coord in filtered
    ])


def parse_selector(selector_str: str) -> dict[Axes, np.ndarray]:
    """Parses a selector string into a dictionary of axes to values"""
    return dict(map(_parse_field_selector(None), selector_str.split(FIELD_DELIM)))  # type: ignore


def select_arr(arr: xr.DataArray, selector_str: str) -> xr.DataArray:
    """Selects a subarray from the given array using provided selector"""
    selector = dict(map(_parse_field_selector(arr), selector_str.split(FIELD_DELIM)))  # type: ignore
    return arr.sel(selector)
