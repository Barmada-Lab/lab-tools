import pathlib as pl

from cvat_sdk import make_client, Client
from skimage.measure import regionprops
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import click 

from ..settings import settings
from .upload import prep_experiment

def rle_to_mask(rle: list[int], width: int, height:int)->np.ndarray:
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

# creates one-to-one mapping of nuclei to soma, based on maximum overlap
def colocalize_rois(nuc_rois, soma_rois):
    for nuc_id in np.unique(nuc_rois[np.nonzero(nuc_rois)]):
        nuc_mask = nuc_rois == nuc_id
        soma_id_contents = soma_rois[nuc_mask][np.nonzero(soma_rois[nuc_mask])]
        soma_id = np.argmax(np.bincount(soma_id_contents))
        yield (nuc_id, soma_id)

def measure_nuc_cyto_ratio(
        client: Client, 
        project_id: int, 
        collections: dict[str, xr.DataArray],
        nuc_channel: str, 
        soma_channel: str,
        measurement_channels: list[str]):

    df = pd.DataFrame()
    for task_name, frame_names, labelled_arr in enumerate_rois(client, project_id):
        ### BEGIN MANUALLY EDITABLE SECTION

        tokens = task_name.split("_")
        # need different formatting depending on experiment........
        region = "_".join(tokens[:-1])
        collection_name = region
        collection = collections[collection_name]
        field = tokens[-1]
        # collection_name = region
        intensity_arr = collection.sel(region=region, field=int(field))

        ### END MANUALLY EDITABLE SECTION

        channels = [name.split(".")[0].split("_")[-1] for name in frame_names]
        nuc_idx = channels.index(nuc_channel)
        soma_idx = channels.index(soma_channel)

        soma_mask = labelled_arr[soma_idx]
        nuclear_mask = labelled_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8) 

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append({
                "id": props.label,
                "area": props.area,
            })
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append({
                "id": props.label,
                "area": props.area,
            })
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:
            field_intensity_arr = intensity_arr.sel(channel=channel).values

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                cyto_df.loc[cyto_df["id"] == props.label, f"{channel}_intensity_sum"] = props.image_intensity.sum()
                cyto_df.loc[cyto_df["id"] == props.label, f"{channel}_intensity_mean"] = props.image_intensity.mean()

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                nuc_df.loc[nuc_df["id"] == props.label, f"{channel}_intensity_sum"] = props.image_intensity.sum()
                nuc_df.loc[nuc_df["id"] == props.label, f"{channel}_intensity_mean"] = props.image_intensity.mean()

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id", suffixes=("_nuc", "_cyto"))
        merged.insert(0, "collection", collection_name)
        df = pd.concat((df, merged))

    return df
    
@click.command("nuc-cyto")
@click.argument("project_name", type=str)
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.argument("nuc_channel", type=str)
@click.argument("soma_channel", type=str)
@click.option("--channels", type=str, default="", help="comma-separated list of channels to measure from")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--experiment-type", type=click.Choice(["legacy", "legacy-icc", "lux", "nd2s"]), default="lux", help="experiment type")
def cli_entry(
    project_name: str, 
    experiment_base: pl.Path, 
    nuc_channel: str,
    soma_channel: str,
    channels: str,
    mip: bool,
    experiment_type: str):

    with make_client(
        host=settings.cvat_url,
        credentials=(
            settings.cvat_username,
            settings.cvat_password
        )
    ) as client:
        org_slug = settings.cvat_org_slug
        client.organization_slug = org_slug
        api_client = client.api_client

        (data, _) = api_client.projects_api.list(search=project_name)
        assert data is not None and len(data.results) > 0, \
            f"No project matching {project_name} in organization {org_slug}"

        try:
            # exact matches only
            project = next(filter(lambda x: x.name == project_name, data.results))
        except StopIteration:
            raise ValueError(f"No project matching {project_name} in organization {org_slug}")

        channel_list = channels.split(",")

        project_id = project.id
        output_dir = experiment_base / "results"
        output_dir.mkdir(exist_ok=True)

        # TODO: homogenize collections and put into one array
        if experiment_type == "nd2s":
            collections = {nd2_file.name.replace(".nd2",""): prep_experiment(nd2_file, mip, False, experiment_type, 0.0, None, False) for nd2_file in experiment_base.glob("**/*.nd2")}
        else:
            collections = {experiment_base.name: prep_experiment(experiment_base, mip, False, experiment_type, 0.0, None, False)}

        df = measure_nuc_cyto_ratio(client, project_id, collections, nuc_channel, soma_channel, channel_list)
        df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)