from collections import defaultdict
from re import error
from typing import Any, Iterable, Type
import dask.array as da

from numpy import indices
from tifffile import tifffile
from .experiments.experiment import Experiment

from pathlib import Path

from tqdm import tqdm
import zarr
import numpy as np

from .experiments import tags, dataset
from .experiments.image import Image
from . import improc_json
from . import arr_experiment as exp

from multiprocessing import Pool
from itertools import chain

def _write_arr(group: zarr.Group, vertex: str, images: list[Image]):
    # ahahahahahahahaha good luck
    #
    #
    def is_timeseries(img: Image):
        return img.get_tag(tags.Timepoint) is not None

    def is_mosaic(img: Image):
        return img.get_tag(tags.MosaicTile) is not None

    def is_zslice(img: Image):
        return img.get_tag(tags.ZSliceMeta) is not None

    def extract_channel(img: Image) -> Any:
        tag = img.get_tag(tags.Channel)
        if tag is not None:
            return tag
        else:
            raise Exception()

    dataset_attrs = {}
    indexing = []
    if all(map(is_timeseries, images)):
        indexing.append((tags.Timepoint, tags.StackIndex.T))
        dataset_attrs["timeseries"] = True
    if all(map(is_mosaic, images)):
        indexing.append((tags.MosaicTile, tags.StackIndex.M))
        dataset_attrs["mosaic"] = images[0].get_tag(tags.Mosaic)
    if all(map(is_zslice, images)):
        indexing.append((tags.ZSliceMeta, tags.StackIndex.Z))
        dataset_attrs["zstack"] = True
    indexing.append((tags.Channel, tags.StackIndex.C))

    channels = list(set(map(extract_channel, images)))
    channel_order = list(tags.Channel)
    # sort channels by precedence in the tags.Channel Enum
    channels = sorted(channels, key=channel_order.index)

    stack_indices = []
    for image in images:
        index = []
        for tag, _ in indexing:
            match tag:
                case tags.Timepoint:
                    tp = image.get_tag(tags.Timepoint)
                    index.append(tp.index)
                case tags.MosaicTile:
                    m = image.get_tag(tags.MosaicTile)
                    index.append(m.index)
                case tags.ZSliceMeta:
                    z = image.get_tag(tags.ZSliceMeta)
                    index.append(z.index)
                case tags.Channel:
                    c = image.get_tag(tags.Channel)
                    index.append(channels.index(c))
        stack_indices.append(index)

    sorted_indices = sorted(zip(images,stack_indices), key=lambda t: t[1])
    guinneapig = sorted_indices[0][0].load()
    try:
        dimx, dimy = guinneapig.shape
    except Exception:
        raise Exception("THEY AINT 2D")

    shape = [i + 1 for i in sorted_indices[-1][1]]
    shape = shape[:-1] + [dimx, dimy] + [shape[-1]] # damn
    chunks = [1 for _ in range(len(shape))]
    chunks[-3:] = shape[-3:]
    dataset = group.create_dataset(name=vertex, shape=shape, chunks=tuple(chunks), dtype="u2")
    inc = 0
    for image, index in tqdm(sorted_indices, desc=f"populating {vertex}", position=1, leave=False):
        idx = tuple(index[:-1] + [slice(None), slice(None), index[-1]]) # oof
        dataset[idx] = image.load()
        inc += 1

    indexing = [idx for _, idx in indexing]
    dataset.attrs["axes"] = [exp.Axis.T, exp.Axis.M, exp.Axis.X, exp.Axis.Y, exp.Axis.C]
    dataset.attrs["channel_meta"] = exp.ChannelMeta([exp.Channel.GFP])
    dataset.attrs["planar_meta"] = exp.PlanarMeta(0.5)

def zarrify(experiment: Experiment, output_path: Path):
    improc_json.monkeypatch_global_encoder()
    group = zarr.open_group(output_path, mode="w")
    raw_images = group.create_group("raw_images", overwrite=True)
    image_sets = dataset.aggregate(experiment.raw, tags.Vertex)
    with Pool() as p:
        for vertex, images in tqdm(image_sets.items(), desc=f"creating zarr", position=0):
            _write_arr(raw_images, vertex.label, images)
    improc_json.restore_original_encoder()

def load(zarr_path: Path):
    g = zarr.open_group(zarr_path, mode="rw")
    return exp.Experiment(zarr_path.name, g)

def arr_iter(a: da.Array, axes: list[int], indices):
    if len(axes) == 0:
        return [(indices, a)]
    else:
        da_iter = da.rollaxis(a, axes[0]-len(indices))
        def branch(args):
            idx: int = args[0]
            o = args[1]
            return arr_iter(o, axes[1:], indices + (idx,))
        return chain.from_iterable(map(branch, enumerate(da_iter)))
