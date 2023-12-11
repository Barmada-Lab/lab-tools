import pathlib as pl

from cvat_sdk import make_client, Client
from tqdm import tqdm
from skimage.measure import regionprops
import pandas as pd
import xarray as xr
import numpy as np
import click

from ..settings import settings
from .upload import prep_experiment

import pathlib as pl

import click

def rle_to_mask(rle: list[int], width: int, height:int) -> np.ndarray:
    decoded = [0] * (width * height) # create bitmap container
    decoded_idx = 0
    value = 0

    for v in rle:
        decoded[decoded_idx:decoded_idx+v] = [value] * v
        decoded_idx += v
        value = abs(value - 1)

    decoded = np.array(decoded, dtype=bool)
    decoded = decoded.reshape((height, width)) # reshape to image size
    return decoded

def get_labelled_arr(anno_table, length, height, width):
    channel_stack_mask = np.zeros((length, height, width), dtype=int)
    for shape in anno_table.shapes:
        id = shape.id
        frame = shape.frame
        rle = list(map(int, shape.points))
        left, top, right, bottom = rle[-4:]
        patch_height, patch_width = (bottom - top + 1, right - left +1)
        patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height).astype(int) * id
        channel_stack_mask[frame, top:bottom+1, left:right+1] += patch_mask 
    return channel_stack_mask

def enumerate_rois(client: Client, project_id: int):
    tasks = client.projects.retrieve(project_id).get_tasks()
    for task_meta in tqdm(tasks):
        jobs = task_meta.get_jobs()
        job_id = jobs[0].id
        job_metadata, _ = client.api_client.jobs_api.retrieve_data_meta(job_id)
        frames = job_metadata.frames
        frame_names = [frame.name for frame in frames]
        width = frames[0].width
        height = frames[0].height
        anno_table = task_meta.get_annotations()
        labelled_arr = get_labelled_arr(anno_table, len(frames), height, width)
        task_name = task_meta.name
        yield task_name, frame_names, labelled_arr

def measure_2d(
        client: Client, 
        project_id: int, 
        collections: dict[str, xr.DataArray], 
        measurement_channels: list[str]):

    df = pd.DataFrame()
    for task_name, _, labelled_arr in enumerate_rois(client, project_id):
        tokens = task_name.split("_")
        collection_name = "_".join(tokens[:-1]) + ".nd2"
        collection = collections[collection_name]
        intensity_arr = collection.sel(field=task_name)
        mask = labelled_arr[0]

        field_measurements = []
        for props in regionprops(mask):
            field_measurements.append({
                "id": props.label,
                "collection": collection_name,
                "area": props.area,
            })
        field_df = pd.DataFrame.from_records(field_measurements)
        
        for channel in measurement_channels:
            field_intensity_arr = intensity_arr.sel(channel=channel).values

            for props in regionprops(mask, intensity_image=field_intensity_arr):
                field_df.loc[field_df["id"] == props.label, f"{channel}_intensity_sum"] = props.image_intensity.sum()
                field_df.loc[field_df["id"] == props.label, f"{channel}_intensity_std"] = props.image_intensity.std()
        
        df = pd.concat((df, field_df))
    
    return df

@click.command("measure")
@click.argument("project_name", type=str)
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.option("--channels", type=str, default="", help="comma-separated list of channels to measure")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["XY", "TXY", "CXY", "ZXY"]), default="XY", help="dims of uploaded stacks")
@click.option("--experiment-type", type=click.Choice(["lux", "nd2s"]), default="lux", help="experiment type")
def cli_entry(
    project_name: str, 
    experiment_base: pl.Path, 
    channels: str, 
    mip: bool,
    dims: str, 
    experiment_type: str):

    channel_list = channels.split(",")
    if channel_list == [""]:
        raise ValueError("Must provide at least one channel to measure")

    if experiment_type == "nd2s":
        collections = {nd2_file.name: prep_experiment(nd2_file, mip, False, experiment_type, 0.0, None, False) for nd2_file in experiment_base.glob("**/*.nd2")}
    else:
        collections = {experiment_base.name: prep_experiment(experiment_base, mip, False, experiment_type, 0.0, None, False)}

    output_dir = experiment_base / "results"
    output_dir.mkdir(exist_ok=True)
   
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
        assert data is not None and len(data.results) > 0, \
            f"No project matching {project_name} in {org_slug}; create a project in the webapp first"

        project = next(filter(lambda x: x.name == project_name, data.results))
        project_id = project.id

        match dims:
            case "XY":
                df = measure_2d(client, project_id, collections, channel_list)
                df.to_csv(output_dir / "measurements_CVAT.csv", index=False)
                return
            case "TXY":
                raise NotImplementedError("TXY measurements not implemented")
            case "CXY":
                raise NotImplementedError("CXY measurements not implemented")
            case "ZXY":
                raise NotImplementedError("ZXY measurements not implemented")
            case _:
                raise ValueError(f"Unknown dims {dims}")