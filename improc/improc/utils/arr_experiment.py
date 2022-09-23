from typing import Any, Iterable
import enum
from datetime import datetime
from dataclasses import dataclass
import itertools
from itertools import chain

import dask.array as da
import zarr


class Axis(enum.Enum):
    M = "M"
    T = "T"
    Z = "Z"
    X = "X"
    Y = "Y"
    C = "C"

class Channel(enum.Enum):
    GFP = "GFP"
    RFP = "RFP"
    Cy5 = "Cy5"
    DAPI = "DAPI"
    BRIGHTFIELD = "white_light"
    PHASE_NEG = "phase_neg"

@dataclass
class DynamicSlice:
    index: tuple[int, ...]
    arr: da.Array

@dataclass
class TimeMeta:
    index: list[datetime]

@dataclass
class MosaicMeta:
    index: list[tuple[int,int]]
    overlap: float

@dataclass
class ZSliceMeta:
    index: list[float]

@dataclass
class PlanarMeta:
    pixel_size: float

@dataclass
class ChannelMeta:
    index: list[Channel]

@dataclass
class Vertex:
    label: str
    data: zarr.Array

    def _attr(self, label: str):
        return self.data.attrs[label]

    @property
    def axes(self) -> list[Axis]:
        return list(map(Axis, self._attr("axes")))

    def axis_of(self, Axis) -> int | None:
        return self.axes.index(Axis)

    @property
    def planar_meta(self) -> PlanarMeta:
        return PlanarMeta(**self._attr("planar_meta"))

    @property
    def channel_meta(self) -> ChannelMeta:
        return ChannelMeta(**self._attr("channel_meta"))

    @property
    def time_meta(self) -> TimeMeta | None:
        if (meta := self._attr("time_meta")) is not None:
            return TimeMeta(**meta)

    @property
    def mosaic_meta(self) -> MosaicMeta | None:
        if (meta := self._attr("mosaic_meta")) is not None:
            return MosaicMeta(**meta)

    @property
    def zslice_meta(self) -> ZSliceMeta | None:
        if (meta := self._attr("zslice_meta")) is not None:
            return ZSliceMeta(**meta)


@dataclass
class Dataset:
    label: str
    data: zarr.Group

    @property
    def vertices(self) -> Iterable[Vertex]:
        return itertools.starmap(Vertex, self.data.arrays())

@dataclass
class Experiment:
    label: str
    data: zarr.Group

    @property
    def datasets(self) -> dict[str, Dataset]:
        return { name: Dataset(name, data) for name, data in self.data.groups() }
