from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
import enum
from functools import cached_property
import os
import pathlib
from typing import Callable, Type, TypeVar

import numpy as np
from skimage import io
from enum import Enum

import tifffile
import shutil

import zarr
import abc
from improc.experiment.legacy.mfile import MFSpec

from improc.utils.enumero import NaturalOrderStrEnum
from improc.utils import rand
import sys

META_KEY = "meta"
AXES_KEY = "axes"

class Axis(Enum):
    X = 1
    Y = 2
    Z = 3
    T = 4
    C = 5

class Channel(NaturalOrderStrEnum):
    GFP = "GFP"
    RFP = "RFP"
    Cy5 = "Cy5"
    DAPI = "DAPI"
    BRIGHTFIELD = "brightfield"

class Objective(NaturalOrderStrEnum):
    X4 = "4X"
    X10 = "10X"
    X20 = "20X"
    X40 = "40X"
    X60 = "60X"

@dataclass(frozen=True, eq=True)
class Vertex:
    label: str

@dataclass(frozen=True, eq=True)
class Exposure:
    duration: timedelta
    channel: Channel

@dataclass(frozen=True, eq=True)
class Timepoint:
    index: int
    realtime: datetime

@dataclass(frozen=True, eq=True)
class Mosaic:
    index: tuple[int,int]
    overlap: float

@dataclass(frozen=True, eq=True)
class Geometry:
    #z_offset: float
    #TODO x_pos: float
    #TODO y_pos: float
    #TODO pixel_size: float
    objective: Objective

Tag = Vertex | Exposure | Timepoint | Mosaic | Geometry
T = TypeVar("T", Vertex, Exposure, Timepoint, Mosaic, Geometry)

@dataclass
class Image(abc.ABC):

    @property
    @abc.abstractmethod
    def axes(self) -> list[Axis]:
        ...

    @property
    @abc.abstractmethod
    def tags(self) -> list[Tag]:
        ...

    @property
    @abc.abstractmethod
    def data(self) -> np.ndarray:
        ...

    def get_tag(self, tag_type: Type[T]) -> T | None:
        for tag in self.tags:
            if isinstance(tag, tag_type):
                return tag
        return None

    @property
    def vertex(self) -> str:
        vertex_tag = self.get_tag(Vertex)
        if vertex_tag is None:
            raise RuntimeError("WHY DOES THIS IMAGE NOT HAVE A VERTEX TAG?")
        return vertex_tag.label

def get_tag(tag_type: Type[T], image: Image) -> T | None:
        for tag in image.tags:
            if isinstance(tag, tag_type):
                return tag
        return None

@dataclass
class MemoryImage(Image):
    _data: np.ndarray
    _axes: list[Axis]
    _tags: list[Tag]

    @property
    def axes(self) -> list[Axis]:
        return self._axes

    @property
    def tags(self) -> list[Tag]:
        return self._tags

    @property
    def data(self) -> np.ndarray:
        return self._data

@dataclass
class ImageFile(Image):
    path: pathlib.Path
    _tags: list[Tag]

    @property
    def axes(self) -> list[Axis]:
        # TODO try to read a private tag?
        return [Axis.X, Axis.Y] # TODO: this probably won't hold up...

    @property
    def tags(self) -> list[Tag]:
        return self._tags

    @property
    def data(self) -> np.ndarray:
        return tifffile.imread(self.path)

@dataclass
class Dataset(abc.ABC):
    label: str

    @property
    @abc.abstractmethod
    def images(self) -> list[Image]:
        ...

    @abc.abstractmethod
    def write_image(self, image: np.ndarray, tags: list[Tag], axes: list[Axis]) -> Image:
        ...

    def write_image2(self, image: Image):
        self.write_image(image.data, image.tags, image.axes)

@dataclass
class DirDataset(Dataset):
    path: pathlib.Path

    def parse_nu_tag(self, token: str) -> Tag:
        subtokens = token.split("_")
        if len(subtokens) == 1:
            return Vertex(subtokens[0])
        elif len(subtokens) == 2:
            if subtokens[0].isnumeric():
                tp = int(subtokens[0])
                timestamp = int(subtokens[1])
                return Timepoint(tp, datetime.fromtimestamp(timestamp))
            else:
                channel = Channel(subtokens[0])
                td = timedelta(milliseconds=int(subtokens[1]))
                return Exposure(td, channel)
        elif len(subtokens) == 3:
            x, y, overlap_dec = map(int, subtokens)
            return Mosaic((x,y), float(overlap_dec) / 10)
        else:
            raise Exception(f"Can't parse tag: {token}")


    def parse_nu_tags(self, name: str) -> list[Tag]:
        tokens = name.split("-")
        return list(map(self.parse_nu_tag, tokens))

    def format_nu_tag(self, tag: Tag) -> str | None:
        match tag:
            case Vertex(label):
                return label
            case Exposure(duration, channel):
                ms_dur = int(duration.total_seconds() * 1000)
                return f"{channel.value}_{ms_dur}"
            case Timepoint(timepoint, realtime):
                ts = int(realtime.timestamp())
                return f"{timepoint}_{ts}"
            case Mosaic(index, overlap):
                return f"{index[0]}_{index[1]}_{int(overlap*10)}"
        return None

    def format_nu_tags(self, tags: list[Tag]) -> str:
        formatted = [tag for tag in map(self.format_nu_tag, tags) if tag is not None]
        return "-".join(formatted)

    @property
    def images(self) -> list[Image]:
        _images = []
        for path in self.path.glob("*.tif"):
            tags = self.parse_nu_tags(path.name.strip(".tif"))
            _images.append(ImageFile(path, tags))
        return _images

    def write_image(self, image: np.ndarray, tags: list[Tag], axes: list[Axis]) -> Image:
        filename = self.format_nu_tags(tags) + ".tif"
        path = self.path / filename
        tifffile.imwrite(path, image)
        return ImageFile(path, tags)


@dataclass
class Experiment:
    label: str
    datasets: dict[str, Dataset]

    experiment_dir: pathlib.Path
    scratch_dir: pathlib.Path

    def new_dataset(self, dataset_label: str, overwrite=False) -> Dataset:
        if dataset_label in self.datasets and not overwrite:
            raise Exception(f"{dataset_label} already exists in experiment; set overwrite=True if you wish to replace this dataset")
        path = self.scratch_dir / dataset_label
        if path.exists() and any(path.iterdir()) and overwrite:
            shutil.rmtree(path)
        elif path.is_file():
            os.remove(path)
        os.makedirs(path, exist_ok=True)
        dataset = DirDataset(dataset_label, path)
        self.datasets[dataset_label] = dataset
        return dataset
