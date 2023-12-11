import pathlib as pl
from itertools import product

import dask.array as da
import xarray as xr
import numpy as np

from ..experiment import Axes
from . import ioutils

def load_legacy(base: pl.Path, fillna: bool = True) -> xr.Dataset:
    timepoint_tags = sorted({int(path.name.replace("T","")) for path in base.glob("raw_imgs/*/*")})
    region_tags = set()
    field_id_tags = set()
    channel_tags = set()
    for path in base.glob("raw_imgs/**/*.tif"):
        channel_tags.add(path.parent.parent.parent.name)
        region, field = path.name.split(".")[0].split("_")
        region_tags.add(region)
        field_id_tags.add(field)
    
    max_field_id = max(map(int, field_id_tags))
    dim = np.sqrt(max_field_id).astype(int)
    field_tags = list(map(lambda x: "_".join(map(str, x)), product(range(dim), range(dim))))

    region_tags = sorted(region_tags)
    channel_tags = sorted(channel_tags)
    field_id_tags = sorted(field_id_tags)
    timepoint_tags = sorted(timepoint_tags)

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT

    channels = []
    for channel in channel_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            regions = []
            for region in region_tags:
                fields = []
                for field in field_id_tags:
                    col = region[1:]
                    path = base / "raw_imgs" / channel / f"T{timepoint}" / f"col_{col}" / f"{region}_{field}.tif"
                    img = ioutils.read_tiff_toarray(path) 
                    fields.append(img)
                regions.append(da.stack(fields))
            timepoints.append(da.stack(regions))
        channels.append(da.stack(timepoints))
    plate = da.stack(channels)

    dataset = xr.Dataset(
        data_vars=dict(
            intensity = xr.DataArray(
                plate,
                dims=[Axes.CHANNEL, Axes.TIME, Axes.REGION, Axes.FIELD, Axes.Y, Axes.X],
                coords={
                    Axes.CHANNEL: channel_tags,
                    Axes.TIME: timepoint_tags,
                    Axes.REGION: region_tags,
                    Axes.FIELD: field_tags,
                }
            ).chunk({
                Axes.CHANNEL: -1, 
                Axes.TIME: -1, 
                Axes.REGION: 1, 
                Axes.FIELD: 1, 
                Axes.Y: -1, 
                Axes.X: -1
            })
        )
    )

    if fillna:
        dataset = dataset.ffill("t").bfill("t").ffill("field").bfill("field")

    return dataset

def load_legacy_icc(base_path: pl.Path, fillna: bool = True) -> xr.Dataset:
    experiment = load_legacy(base_path, fillna=fillna).squeeze(Axes.REGION, drop=True)
    renamed = experiment.rename({Axes.TIME: Axes.REGION})
    return renamed.assign_coords(
            {Axes.REGION: renamed[Axes.REGION].astype(str)}
        ).chunk({
            Axes.CHANNEL: -1, 
            Axes.REGION: 1, 
            Axes.FIELD: 1, 
            Axes.Y: -1, 
            Axes.X: -1
        })