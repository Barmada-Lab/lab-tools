import warnings
from itertools import product
import re
import pathlib as pl

from skimage import exposure # type: ignore
import xarray as xr
import numpy as np
import dask.array as da
import dask
import ome_types
import tifffile
import dask
import nd2

wellplate_96_names = [
    "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12",
    "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12",
    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11", "D12",
    "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12",
    "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11", "F12",
    "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12",
]

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
    loc_tags = set()
    field_tags = set()
    exposure_tags = set()
    for path in base.glob("raw_imgs/*/*.tif"):
        loc, field, exposure = path.name.split(".")[0].split("-")
        loc_tags.add(loc)
        field_tags.add(field)
        exposure_tags.add(exposure)

    loc_tags = sorted(loc_tags)
    field_tags = sorted(field_tags)
    exposure_tags = sorted(exposure_tags)
    timepoint_tags = sorted(timepoint_tags)

    test_path = base / f"raw_imgs/T{timepoint_tags[0]}" / f"{loc_tags[0]}-{field_tags[0]}-{exposure_tags[0]}.tif"
    test_img = tifffile.imread(test_path)
    shape = test_img.shape

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT
    channels = []
    for exposure in exposure_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            locs = []
            for loc in loc_tags:
                fields = []
                for field in field_tags:
                    path = base / "raw_imgs" / f"T{timepoint}" / f"{loc}-{field}-{exposure}.tif"
                    img = read_tiff_toarray(path, shape)
                    fields.append(img)
                locs.append(da.stack(fields))
            timepoints.append(da.stack(locs))
        channels.append(da.stack(timepoints).rechunk((-1,1,1,-1,-1)))
    plate = da.stack(channels)

    loc_names = [loc.replace("well_","") for loc in loc_tags]
    loc_coords = list(map(wellplate_96_names.index, loc_names))
    field_coords = [field.replace("mosaic_","") for field in field_tags]
    channel_coords = [exposure.split("_")[0] for exposure in exposure_tags]

    dataset = xr.Dataset(
        data_vars=dict(
            intensity = xr.DataArray(
                plate,
                dims=("channel", "t", "loc", "field", "y", "x"),
                coords={
                    "channel": channel_coords,
                    "t": timepoint_tags,
                    "loc": loc_coords,
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
    loc_tags = set()
    field_id_tags = set()
    channel_tags = set()
    for path in base.glob("raw_imgs/**/*.tif"):
        channel_tags.add(path.parent.parent.parent.name)
        loc, field = path.name.split(".")[0].split("_")
        loc_tags.add(loc)
        field_id_tags.add(field)
    
    max_field_id = max(map(int, field_id_tags))
    dim = np.sqrt(max_field_id).astype(int)
    field_tags = list(map(lambda x: "_".join(map(str, x)), product(range(dim), range(dim))))

    loc_tags = sorted(loc_tags)
    channel_tags = sorted(channel_tags)
    field_id_tags = sorted(field_id_tags)
    timepoint_tags = sorted(timepoint_tags)

    test_path = next(base.glob("raw_imgs/**/*.tif"))
    test_img = tifffile.imread(test_path)
    shape = test_img.shape

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT

    channels = []
    for channel in channel_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            locs = []
            for loc in loc_tags:
                fields = []
                for field in field_id_tags:
                    col = loc[1:]
                    path = base / "raw_imgs" / channel / f"T{timepoint}" / f"col_{col}" / f"{loc}_{field}.tif"
                    img = read_tiff_toarray(path, shape) 
                    fields.append(img)
                locs.append(da.stack(fields))
            timepoints.append(da.stack(locs))
        channels.append(da.stack(timepoints).rechunk((-1,1,1,-1,-1)))
    plate = da.stack(channels)

    dataset = xr.Dataset(
        data_vars=dict(
            intensity = xr.DataArray(
                plate,
                dims=["channel", "t", "loc", "field", "y", "x"],
                coords={
                    "channel": channel_tags,
                    "t": timepoint_tags,
                    "loc": loc_tags,
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
    return experiment.squeeze(drop=True).rename(t="loc")

def read_nd2(base_path: pl.Path):

    arr = nd2.imread(base_path, dask=True, xarray=True)
    nd2_label = base_path.name.replace(".nd2","")
    arr = arr.expand_dims("loc").assign_coords(loc=[nd2_label])

    # single-channel images don't include C
    if "C" not in arr.dims:
        arr = arr.expand_dims("C")
        channel = arr.metadata["metadata"].channels[0].channel.name.strip()
        arr = arr.assign_coords(C=[channel])
    else:
        # sanitize inputs that may contain leading/trailing spaces
        arr = arr.assign_coords(C=[channel.strip() for channel in arr.C.values])
    # single-field images don't include P
    if "P" not in arr.dims:
        arr = arr.expand_dims("P")
        point_coords = [f"0"]
    else:
        point_coords = list(range(arr.P.size))
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

def read_series(img):
    try:
        arr = img.asarray()
        return exposure.rescale_intensity(arr, out_range=np.float32)
    except (ValueError, NameError, FileNotFoundError) as e:
        warnings.warn(f"Error reading {img.name}: {e}\nThis field will be filled based on surrounding fields and timepoints.")
        arr = np.zeros(img.shape, dtype=np.float32)
        arr[:] = np.nan
        return arr


def read_cq1_experiment(base_path: pl.Path):

    ome_tiff = base_path / "MeasurementResults.ome.tif"
    with tifffile.TiffFile(ome_tiff) as tiff:

        if tiff.ome_metadata is None:
            raise ValueError(f"Missing OME metadata in {ome_tiff}")

        acquisition_meta = ome_types.from_xml(tiff.ome_metadata)

        # Given a string like W14(C2R2),A1,F1, picks out the loc index (14) and field idx
        regex = re.compile(
            r"W(?P<loc_idx>\d+)\,A(?P<unknown>\d+)\,F(?P<field_idx>\d+)")

        for series in tiff.series[1:]: # skip the first image, which is metadata

            if series.name is None: 
                raise ValueError(f"Img is missing name: {series}")

            search = regex.search(series.name)

            if search is None: 
                raise ValueError(f"Couldn't find loc index in {series.name}")

            loc_idx = int(search.group("loc_idx")) - 1 # 1-indexed in the name, 0-indexed in the array
            field_idx = int(search.group("field_idx")) - 1

            try:
                img_meta = next(img for img in acquisition_meta.images if img.name == series.name)
            except StopIteration:
                raise ValueError(f"Missing metadata for {series.name}")
            
            channels = [channel.name for channel in img_meta.pixels.channels]
            axes = [ax for ax in series.axes]

            arr = xr.DataArray(
                da.from_delayed(
                    dask.delayed(read_series)(series), series.shape, dtype=np.float32),
                dims=axes,
                coords={
                    "channel": channels
                }
            )
