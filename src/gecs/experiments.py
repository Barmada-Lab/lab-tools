import warnings
from itertools import product
import pathlib as pl

from skimage import exposure # type: ignore
import xarray as xr
import numpy as np
import dask.array as da
import dask
import tifffile
import dask
import nd2

def read_tiff_delayed(shape: tuple):
    def read(path: pl.Path) -> np.ndarray:
        try:
            img = tifffile.imread(path)
            return exposure.rescale_intensity(img, out_range=np.float32)
        except (ValueError, NameError, FileNotFoundError) as e:
            warnings.warn(f"Error reading {path}: {e}\nThis field will be filled based on surrounding fields and timepoints.")
            img = np.zeros(shape, dtype=np.float32)
            img[:] = np.nan
            return img
        
    return dask.delayed(read)

def read_tiff_toarray(path: pl.Path, shape: tuple):
    return da.from_delayed(read_tiff_delayed(shape)(path), shape, dtype=np.float32)

def read_lux_experiment(base: pl.Path, fillna: bool = True):
    timepoint_tags = sorted([int(path.name.replace("T","")) for path in base.glob("raw_imgs/*")])
    well_tags = set()
    field_tags = set()
    exposure_tags = set()
    for path in base.glob("raw_imgs/*/*.tif"):
        well, field, exposure = path.name.split(".")[0].split("-")
        well_tags.add(well)
        field_tags.add(field)
        exposure_tags.add(exposure)

    well_tags = sorted(well_tags)
    field_tags = sorted(field_tags)
    exposure_tags = sorted(exposure_tags)
    timepoint_tags = sorted(timepoint_tags)

    path = base / f"raw_imgs/T{timepoint_tags[0]}" / f"{well_tags[0]}-{field_tags[0]}-{exposure_tags[0]}.tif"
    test_img = tifffile.imread(path)
    shape = test_img.shape

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT
    channels = []
    for exposure in exposure_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            wells = []
            for well in well_tags:
                fields = []
                for field in field_tags:
                    path = base / "raw_imgs" / f"T{timepoint}" / f"{well}-{field}-{exposure}.tif"
                    img = read_tiff_toarray(path, shape)
                    fields.append(img)
                wells.append(da.stack(fields))
            timepoints.append(da.stack(wells))
        channels.append(da.stack(timepoints).rechunk((-1,1,1,-1,-1)))
    plate = da.stack(channels)

    well_coords = [well.replace("well_","") for well in well_tags]
    field_coords = [field.replace("mosaic_","") for field in field_tags]
    channel_coords = [exposure.split("_")[0] for exposure in exposure_tags]

    dataset = xr.Dataset(
        data_vars=dict(
            intensity = xr.DataArray(
                plate,
                dims=("channel", "t", "well", "field", "y", "x"),
                coords={
                    "channel": channel_coords,
                    "t": timepoint_tags,
                    "well": well_coords,
                    "field": field_coords,
                }
            )
        )
    )

    if fillna:
        return dataset.ffill("t").bfill("t").ffill("field").bfill("field")
    else:
        return dataset

def read_legacy_experiment(base: pl.Path, fillna: bool = True):
    timepoint_tags = sorted({int(path.name.replace("T","")) for path in base.glob("raw_imgs/*/*")})
    well_tags = set()
    field_id_tags = set()
    channel_tags = set()
    for path in base.glob("raw_imgs/**/*.tif"):
        channel_tags.add(path.parent.parent.parent.name)
        well, field = path.name.split(".")[0].split("_")
        well_tags.add(well)
        field_id_tags.add(field)
    
    max_field_id = max(map(int, field_id_tags))
    dim = np.sqrt(max_field_id).astype(int)
    field_tags = list(map(lambda x: "_".join(map(str, x)), product(range(dim), range(dim))))

    well_tags = sorted(well_tags)
    channel_tags = sorted(channel_tags)
    field_id_tags = sorted(field_id_tags)
    timepoint_tags = sorted(timepoint_tags)

    path = next(base.glob("raw_imgs/**/*.tif"))
    test_img = tifffile.imread(path)
    shape = test_img.shape

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT

    channels = []
    for channel in channel_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            wells = []
            for well in well_tags:
                fields = []
                for field in field_id_tags:
                    col = well[1:]
                    path = base / "raw_imgs" / channel / f"T{timepoint}" / f"col_{col}" / f"{well}_{field}.tif"
                    img = read_tiff_toarray(path, shape) 
                    fields.append(img)
                wells.append(da.stack(fields))
            timepoints.append(da.stack(wells))
        channels.append(da.stack(timepoints).rechunk((-1,1,1,-1,-1)))
    plate = da.stack(channels)

    dataset = xr.Dataset(
        data_vars=dict(
            intensity = xr.DataArray(
                plate,
                dims=["channel", "t", "well", "field", "y", "x"],
                coords={
                    "channel": channel_tags,
                    "t": timepoint_tags,
                    "well": well_tags,
                    "field": field_tags,
                }
            )
        )
    )

    if fillna:
        return dataset.ffill("t").bfill("t").ffill("field").bfill("field")
    else:
        return dataset

def read_legacy_icc_experiment(base_path: pl.Path, fillna: bool = True):
    experiment = read_legacy_experiment(base_path, fillna=fillna)
    return experiment.squeeze(drop=True).rename(t="well")

def read_nd2(base_path: pl.Path):

    arr = nd2.imread(base_path, dask=True, xarray=True)
    nd2_label = base_path.name.replace(".nd2","")
    # single-channel images don't include C
    if "C" not in arr.dims:
        arr = arr.expand_dims("C")
        channel = arr.metadata["metadata"].channels[0].channel.name
        arr = arr.assign_coords(C=[channel])
    # single-field images don't include P
    if "P" not in arr.dims:
        arr = arr.expand_dims("P")
        point_coords = [f"{nd2_label}_0"]
    else:
        point_coords = [f"{nd2_label}_{label.split(':')[-1]}" for label in arr.P.values]
    arr = arr.assign_coords(P=point_coords)
    rename_dict = dict(
        C="channel",
        P="field",
        Y="y",
        X="x",
    )
    if "T" in arr.dims:
        rename_dict["T"] = "t"
    if "Z" in arr.dims:
        rename_dict["Z"] = "z"
    arr = arr.rename(rename_dict)

    return xr.Dataset(dict(intensity=arr))