from itertools import product
import warnings
import time
from typing import Callable
import pathlib as pl
import tempfile 
import random

from cvat_sdk import make_client 
from cvat_sdk.models import TaskWriteRequest
from dask.distributed import Client, wait
from toolz import curry
import xarray as xr
import tifffile
import click
import tqdm

from .. import experiments
from .. import display
from ..settings import settings

@curry
def stage_single_frame(arr: xr.DataArray, label: str, tmpdir: pl.Path) -> list[pl.Path]:
    outpath = pl.Path(tmpdir) / f"{label}.tif"
    tifffile.imwrite(outpath, arr)
    return [outpath]

@curry
def stage_t_stack(arr: xr.DataArray, label: str, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for t in arr.t:
        frame = arr.sel(t=t)
        outpath = pl.Path(tmpdir) / f"{label}_{t.values}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images

@curry
def stage_channel_stack(arr: xr.DataArray, label:str, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for c in arr.channel:
        frame = arr.sel(channel=c)
        outpath = pl.Path(tmpdir) / f"{label}_{c.values}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images

@curry
def stage_z_stack(arr: xr.DataArray, label:str, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for z in arr.channel:
        frame = arr.sel(z=z)
        outpath = pl.Path(tmpdir) / f"{label}_{z.values}.tif"
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
        stage_arr: Callable[[str, pl.Path], list[pl.Path]]):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pl.Path(tmpdir)
        images = stage_arr(label, tmpdir)
        upload(client, project_id, label, images)

def prep_experiment(
        experiment_base: pl.Path, 
        mip: bool, 
        composite: bool, 
        experiment_type: str, 
        rescale: float, 
        channels: list[str] | None, 
        apply_psuedocolor: bool = True,
        to_8bit: bool = True,
        dims: str = "XY"):
    match experiment_type:
        case "legacy":
            experiment = experiments.read_legacy_experiment(experiment_base).intensity
        case "legacy-icc":
            experiment = experiments.read_legacy_icc_experiment(experiment_base, fillna=False).intensity
        case "lux":
            experiment = experiments.read_lux_experiment(experiment_base).intensity
        case "nd2s":
            experiment = experiments.read_nd2(experiment_base).intensity
        case _:
            raise ValueError(f"Unknown experiment type {experiment_type}")

    attrs = experiment.attrs

    if channels is not None:
        experiment = experiment.sel(channel=channels)

    if mip:
        if "z" not in experiment.dims:
            raise ValueError("MIP requested but no z-dimension found")
        experiment = experiment.max(dim="z")

    if apply_psuedocolor:
        experiment = display.apply_psuedocolor(experiment.assign_attrs(attrs))

    if composite:
        if "channel" not in experiment.dims:
            warnings.warn("Composite requested but no channel dimension found; ignoring")
        experiment = experiment.mean(dim="channel")

    experiment = display.rescale_intensity(experiment, ["y", "x"], in_percentile=(rescale, 100-rescale), out_range="uint8")
    return experiment

@click.command("cvat-upload-experiment")
@click.argument("project_name", type=str)
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.option("--channels", type=str, default="", help="comma-separated list of channels to include")
@click.option("--composite", is_flag=True, default=False, help="composites channels if set, else uploads each channel separately")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["XY", "TXY", "CXY", "ZXY"]), default="XY", help="dims of uploaded stacks")
@click.option("--experiment-type", type=click.Choice(["legacy", "legacy-icc", "lux", "nd2s"]), default="lux", help="experiment type")
@click.option("--rescale", type=float, default=0.1, 
              help="rescales images by stretching the range of their values to be bounded by the given percentile range, e.g. a value of 1 will rescale an image so that 0 1st percentile and 255 is the 99th percentile")
@click.option("--samples-per-well", type=int, default=-1, help="number of fields to upload per well")
def cli_entry(
    project_name: str, 
    experiment_base: pl.Path, 
    channels: str, 
    composite: bool,
    mip: bool,
    dims: str, 
    experiment_type: str,
    rescale: float,
    samples_per_well: int):

    dask_client = Client(n_workers=1)
    print(dask_client.dashboard_link)

    channel_list = None if channels == "" else channels.split(",")

    if experiment_type == "nd2s":
        collections = [prep_experiment(nd2_file, mip, composite, experiment_type, rescale, channel_list, True) for nd2_file in experiment_base.glob("**/*.nd2")]
    else:
        collections = [prep_experiment(experiment_base, mip, composite, experiment_type, rescale, channel_list, True)]
        
    with make_client(
        host=settings.cvat_url,
        credentials=(
            settings.cvat_username,
            settings.cvat_password
        )
    ) as client:

        client.organization_slug = settings.cvat_org_slug
        
        (data, _) = client.api_client.projects_api.list(search=project_name)
        assert data is not None and len(data.results) > 0, \
            f"No project matching {project_name} in {org_slug}; create a project in the webapp first"

        project = next(filter(lambda x: x.name == project_name, data.results))
        project_id = project.id

        for collection in collections:
            match dims:
                case "XY":
                    assert {*collection.dims} == {"field", "channel", "x", "y", "rgb"}, collection.dims
                    sample = collection.field if samples_per_well == -1 else random.sample([field for field in collection.field], samples_per_well)
                    for field in sample:
                        arr = collection.sel(field=field)
                        label = f"{field.values}"
                        stage_and_upload(client, project_id, label, stage_single_frame(arr)) # type: ignore

                case "TXY":
                    collection = collection.squeeze()
                    assert {*collection.dims} == {"well", "field", "t", "x", "y", "rgb"}, collection.dims
                    for well in collection.well:
                        well_arr = collection.sel(well=well).load()
                        sample = collection.field if samples_per_well == -1 else random.sample([field for field in collection.field], samples_per_well)
                        for field in sample:
                            arr = well_arr.sel(field=field)
                            label = f"{well.values}_{field.values}"
                            stage_and_upload(client, project_id, label, stage_t_stack(arr)) # type: ignore

                case "CXY":
                    assert {*collection.dims} == {"field", "channel", "x", "y", "rgb"}, collection.dims
                    sample = collection.field if samples_per_well == -1 else random.sample([field for field in collection.field], samples_per_well)
                    for field in sample:
                        arr = collection.sel(field=field).load()
                        label = f"{field.values}"
                        stage_and_upload(client, project_id, label, stage_channel_stack(arr)) # type: ignore

                case "ZXY":
                    assert {*collection.dims} == {"field", "z", "x", "y", "rgb"}, collection.dims
                    sample = collection.field if samples_per_well == -1 else random.sample([field for field in collection.field], samples_per_well)
                    for field in sample:
                        arr = collection.sel(field=field)
                        label = f"{field.values}"
                        stage_and_upload(client, project_id, label, stage_z_stack(arr)) # type: ignore

                case _:
                    raise ValueError(f"Unknown dims {dims}")
