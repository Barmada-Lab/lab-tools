import pathlib as pl
import socket

import dask
from dask.distributed import Client
import dask.array as da
import xarray as xr
import numpy as np
import tifffile
from dask_jobqueue import SLURMCluster
from PIL import Image
from pystackreg import StackReg
from skimage import segmentation, morphology, exposure, filters # type: ignore

def read_tiff_delayed(shape: tuple, dtype: str | type):
    def read(path: pl.Path) -> np.ndarray:
        try:
            return tifffile.imread(path)
        except (ValueError, NameError, FileNotFoundError):
            nas = np.zeros(shape, dtype=dtype)
            nas[:] = np.nan
            return nas
        
    return dask.delayed(read) # type: ignore

def read_tiff_toarray(path: pl.Path, shape: tuple, dtype: str | type):
    return da.from_delayed(read_tiff_delayed(shape, dtype)(path), shape, dtype=dtype)

def read_lux_experiment(base: pl.Path):
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
    dtype = test_img.dtype

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT
    wells = []
    for well in well_tags:
        fields = []
        for field in field_tags:
            exposures = []
            for exposure in exposure_tags:
                timepoints = []
                for timepoint in timepoint_tags:
                    path = base / f"raw_imgs/T{timepoint}" / f"{well}-{field}-{exposure}.tif"
                    img = read_tiff_toarray(path, shape, dtype) # type: ignore
                    timepoints.append(img)
                exposures.append(da.stack(timepoints).rechunk((-1,-1,-1))) # type: ignore
            fields.append(da.stack(exposures))
        wells.append(da.stack(fields))
    plate = da.stack(wells)

    well_coords = [well.replace("well_","") for well in well_tags]
    field_coords = [field.replace("mosaic_","") for field in field_tags]
    channel_coords = [exposure.split("_")[0] for exposure in exposure_tags]

    return xr.DataArray(
        plate, name="raw",
        dims=("well", "field", "channel", "t", "y", "x"),
        coords={
            "well": well_coords,
            "field": field_coords,
            "channel": channel_coords,
            "t": timepoint_tags
        }
    )

def logmax_filter(
        arr: xr.DataArray, 
        min_sigma: float = 14, 
        max_sigma: float = 25, 
        n_sigma: int = 15):
    def apply(frame):
        padding = int(max_sigma)
        padded = np.pad(frame, padding, mode='edge')
        sigmas = np.linspace(min_sigma, max_sigma, n_sigma)
        filter_stack = np.array(
            [filters.laplace(filters.gaussian(padded, sigma=sigma)) for sigma in sigmas])
        unpadded = filter_stack[:, padding:-padding, padding:-padding]
        filtered = unpadded.max(axis=0)
        return filtered
    return xr.apply_ufunc(
        apply,
        arr,
        input_core_dims=[["y","x"]],
        output_core_dims=[["y","x"]],
        dask="parallelized",
        vectorize=True)

def segment_logmaxed_stack(logmaxed: xr.DataArray, min_dia: int = 12):
    def apply(logmaxed):
        thresh = filters.threshold_otsu(logmaxed)
        mask = logmaxed > thresh
        se = morphology.disk(min_dia // 2)
        opened = np.array(
            [morphology.binary_opening(frame, se) for frame in mask])
        return opened
    return xr.apply_ufunc(
        apply,
        logmaxed,
        input_core_dims=[["t","y","x"]],
        output_core_dims=[["t","y","x"]],
        dask="parallelized",
        vectorize=True)

def register(arr: xr.DataArray):
    def apply(stack):
        sr = StackReg(StackReg.RIGID_BODY)
        return sr.register_stack(stack)
    return xr.apply_ufunc(
        apply,
        arr,
        input_core_dims=[["t","y","x"]],
        output_core_dims=[["t","tmat_y","tmat_x"]],
        dask_gufunc_kwargs=dict(output_sizes={"tmat_y": 3, "tmat_x": 3}),
        dask="parallelized",
        vectorize=True)

def transform(arr, tmats):
    def apply(stack, tmats):
        sr = StackReg(StackReg.RIGID_BODY)
        transformed = sr.transform_stack(stack, tmats=tmats)
        return transformed
    return xr.apply_ufunc(
        apply,
        arr,
        tmats,
        input_core_dims=[["t","y","x"],["t", "tmat_y", "tmat_x"]],
        output_core_dims=[["t","y","x"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        vectorize=True)

def annotate_segmentation(raw, segmented, color=(1,1,0)):
    def apply(raw, segmented):
        rescaled = exposure.rescale_intensity(raw, out_range='uint8')
        marked = segmentation.mark_boundaries(
            rescaled, segmented, color=color)
        marked = exposure.rescale_intensity(marked, out_range='uint8')
        return marked
    return xr.apply_ufunc(
        apply,
        raw,
        segmented,
        input_core_dims=[["y","x"],["y","x"]],
        output_core_dims=[["y","x","rgb"]],
        dask_gufunc_kwargs=dict(output_sizes={"rgb": 3}),
        output_dtypes=[np.uint8],
        dask="parallelized",
        vectorize=True)
    
def write_ts_chunks_as_gifs(arr, base):
    def apply(arr):
        name = "-".join(map(str, [arr.well.values[0], arr.field.values[0], arr.channel.values[0]]))
        path = base / f"{name}.gif"
        data = arr.data.squeeze()
        frame_0 = Image.fromarray(data[0])
        the_rest = [Image.fromarray(frame) for frame in data[1:]]
        frame_0.save(path, format='GIF', save_all=True, 
            append_images=the_rest, duration=500, loop=0)
        return arr
    return xr.map_blocks(apply, arr, template=arr)

def run(path: pl.Path, scratch: pl.Path, use_stored: bool = False):

    cluster = SLURMCluster(
        account="sbarmada0",
        cores=1,
        memory="7 GB",
        processes=1,
        interface="ib0",
        local_directory="/scratch/sbarmada_root/sbarmada0/jwaksmac/",
        scheduler_options={
            "dashboard_address": "0.0.0.0:42613"
        },
    )
    print(cluster.dashboard_link)

    cluster.scale(10)
    client = Client(cluster, scheduler_file=scratch / "scheduler.json")

    if use_stored:
        zarr_path = scratch / "raw.zarr"
        if not zarr_path.exists():
            experiment = read_lux_experiment(path)
            experiment.to_zarr(scratch / "raw.zarr")
        experiment = xr.open_zarr(scratch / "raw.zarr").raw
    else:
        experiment = read_lux_experiment(path)

    experiment.persist()
    survival_marker = experiment.sel(channel="GFP")
    logmax = logmax_filter(survival_marker)
    segmented = segment_logmaxed_stack(logmax)
    tmats = register(logmax)
    raw_registered = transform(experiment, tmats)
    segmented_registered = transform(segmented, tmats) > 0.5
    annotated = annotate_segmentation(raw_registered, segmented_registered)
    annotated.sel()

    audit_output = path / "audited"
    audit_output.mkdir(exist_ok=True)
    write_ts_chunks_as_gifs(annotated, audit_output).compute()