from itertools import product
import warnings
import time
from typing import Callable
import pathlib as pl
import tempfile 
import random

from cvat_sdk import make_client 
from cvat_sdk.models import TaskWriteRequest, ProjectWriteRequest
from dask.distributed import Client, wait
from toolz import curry
import xarray as xr
import tifffile
import click

from gecs.io.legacy_loader import load_legacy, load_legacy_icc
from gecs.io.lux_loader import load_lux
from gecs.io.nd2_loader import load_nd2

from .. import display
from ..settings import settings
from gecs.experiment import Axes, ExperimentType, coord_selector
from gecs.io.loader import load_experiment

@curry
def stage_single_frame(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    selector_label = coord_selector(arr)
    outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
    tifffile.imwrite(outpath, arr)
    return [outpath]

@curry
def stage_t_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for t in arr[Axes.TIME]:
        frame = arr.sel({Axes.TIME: t})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images

@curry
def stage_channel_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for c in arr[Axes.CHANNEL]:
        frame = arr.sel({Axes.CHANNEL: c})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images

@curry
def stage_z_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for z in arr[Axes.Z]:
        frame = arr.sel({Axes.Z: z})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images


def upload(client, project_id: int, label: str, images: list[pl.Path]):
    for i in range(5):
        try:
            client.tasks.create_from_data(
                spec=TaskWriteRequest(
                    name=label,
                    project_id=project_id),
                resources=images,
                data_params=dict(
                    image_quality=100, 
                    sorting_method="predefined"))
            return
        except Exception as e:
            if i == 4:
                print(f"Failed to upload {label} after 5 attempts. Skipping.")
            else:
                print(f"Error uploading {label}: {e}")
                print(f"Retrying in {i ** 2} seconds")
                time.sleep(i ** 2)

def stage_and_upload(
        client, 
        project_id: int, 
        label: str, 
        stage_arr: Callable[[pl.Path], list[pl.Path]]):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pl.Path(tmpdir)
        images = stage_arr(tmpdir)
        upload(client, project_id, label, images)

def prep_experiment(
        experiment_base: pl.Path, 
        mip: bool, 
        composite: bool, 
        experiment_type: ExperimentType, 
        rescale: float, 
        channels: str | list[str] | None, 
        apply_psuedocolor: bool = True,
        to_uint8: bool = True,
        fillna: bool = True):
    
    experiment = load_experiment(experiment_base, experiment_type, fillna).intensity

    attrs = experiment.attrs

    if channels is not None:
        experiment = experiment.sel({Axes.CHANNEL: channels})

    if mip:
        if Axes.Z not in experiment.dims:
            raise ValueError("MIP requested but no z-dimension found")
        experiment = experiment.max(dim=Axes.Z)

    if apply_psuedocolor:
        experiment = display.apply_psuedocolor(experiment.assign_attrs(attrs))

    if composite:
        if Axes.CHANNEL not in experiment.dims:
            warnings.warn("Composite requested but no channel dimension found; ignoring")
        experiment = experiment.mean(dim=Axes.CHANNEL)

    if to_uint8:
        experiment = display.rescale_intensity(experiment, [Axes.Y, Axes.X], in_percentile=(rescale, 100-rescale), out_range="uint8")

    return experiment

@click.command("upload")
@click.argument("project_name", type=str)
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.option("--channels", type=str, default="", help="comma-separated list of channels to include")
@click.option("--tp", type=int, default=-1, help="timepoint to upload; if -1, uploads all timepoints")
@click.option("--composite", is_flag=True, default=False, help="composite channels if set, else uploads each channel separately")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["XY", "TXY", "CXY", "ZXY"]), default="XY", help="dims of uploaded stacks")
@click.option("--experiment-type", type=click.Choice(ExperimentType.__members__), callback=lambda c, p, v: getattr(ExperimentType, v) if v else None, help="experiment type") # type: ignore
@click.option("--rescale", type=float, default=0.1, 
              help="rescales images by stretching the range of their values to be bounded by the given percentile range, e.g. a value of 1 will rescale an image so that 0 1st percentile and 255 is the 99th percentile")
@click.option("--samples-per-region", type=int, default=-1, help="number of fields to upload per region")
@click.option("--no-fillna", is_flag=True, default=False, help="don't interpolate NaNs")
def cli_entry(
    project_name: str, 
    experiment_base: pl.Path, 
    channels: str, 
    tp: int,
    composite: bool,
    mip: bool,
    dims: str, 
    experiment_type: ExperimentType,
    rescale: float,
    samples_per_region: int,
    no_fillna: bool):

    dask_client = Client(n_workers=1)
    print(dask_client.dashboard_link)

    channel_list = None if channels == "" else channels.split(",")
    if channel_list is not None and len(channel_list) == 1:
        channel_list = channel_list[0]

    if experiment_type == "nd2s":
        collections = [prep_experiment(nd2_file, mip, composite, experiment_type, rescale, channel_list, apply_psuedocolor=True, fillna=not no_fillna) for nd2_file in experiment_base.glob("**/*.nd2")]
    else:
        collections = [prep_experiment(experiment_base, mip, composite, experiment_type, rescale, channel_list, apply_psuedocolor=True, fillna=not no_fillna)]
    
        
    with make_client(
        host=settings.cvat_url,
        credentials=(
            settings.cvat_username,
            settings.cvat_password
        )
    ) as client:

        org_slug = settings.cvat_org_slug
        client.organization_slug = org_slug
        
        (data, _) = client.api_client.projects_api.list(search=project_name)
        if data is None or len(data.results) == 0:
            (project, _) = client.api_client.projects_api.create(
                ProjectWriteRequest(name=project_name) # type: ignore
            )

        else:
            project = next(filter(lambda x: x.name == project_name, data.results))

        project_id = project.id

        for collection in collections:
            if tp != -1:
                collection = collection.sel({Axes.TIME: tp})
            print(collection)
            match dims:
                case "XY":
                    assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                    for region in collection[Axes.REGION]:
                        region_arr = collection.sel({Axes.REGION: region})
                        sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                        for field in sample:
                            arr = region_arr.sel({Axes.FIELD: field})
                            selector_label = coord_selector(arr)
                            stage_and_upload(client, project_id, selector_label, stage_single_frame(arr)) # type: ignore

                case "TXY":
                    assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.TIME, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                    for region in collection[Axes.REGION]:
                        sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                        region_arr = collection.sel({Axes.REGION: region})
                        for field in sample:
                            arr = region_arr.sel({Axes.FIELD: field})
                            selector_label = coord_selector(arr)
                            stage_and_upload(client, project_id, selector_label, stage_t_stack(arr)) # type: ignore

                case "CXY":
                    assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.CHANNEL, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                    for region in collection[Axes.REGION]:
                        region_arr = collection.sel({Axes.REGION: region})
                        sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                        for field in sample:
                            arr = region_arr.sel({Axes.FIELD: field})
                            selector_label = coord_selector(arr)
                            stage_and_upload(client, project_id, selector_label, stage_channel_stack(arr)) # type: ignore

                case "ZXY":
                    assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.Z, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                    for region in collection[Axes.REGION]:
                        region_arr = collection.sel({Axes.REGION: region})
                        sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                        for field in sample:
                            arr = region_arr.sel({Axes.FIELD: field})
                            selector_label = coord_selector(arr)
                            stage_and_upload(client, project_id, selector_label, stage_z_stack(arr)) # type: ignore

                case _:
                    raise ValueError(f"Unknown dims {dims}")
