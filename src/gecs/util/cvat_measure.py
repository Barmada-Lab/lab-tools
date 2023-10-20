from typing import Any
import itertools
import pathlib as pl

from cvat_sdk import make_client, Client
from skimage.measure import regionprops
import numpy as np
import napari
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ..settings import settings
from .cvat_upload_experiment import prep_experiment

def measure_diff(mask_channel1: str, mask_channel2: str, measurement_channels: list[str]):
    def _measure_diff(intensity_stack, label_stack, channels, stack_label):
        rows = []
        for channel in measurement_channels:
            for props in regionprops(diff, intensity_image=intensity_stack.sel(channel=channel).values):
                rows.append({
                    "label": stack_label,
                    "channel": channel,
                    "roi_id": props.label,
                    "area": props.area,
                    "integrated_intensity": props.image_intensity.sum()
                })
        return rows
    return _measure_diff

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
    channel_stack_mask = np.zeros((length, height, width), dtype=bool)
    for shape in anno_table.shapes:
        frame = shape.frame
        rle = list(map(int, shape.points))
        left, top, right, bottom = rle[-4:]
        patch_height, patch_width = (bottom - top + 1, right - left +1)
        patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height)
        channel_stack_mask[frame, top:bottom+1, left:right+1] = patch_mask

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
        yield frame_names, labelled_arr

def colocalize_rois(roi_img1, roi_img2) -> list[tuple[int, int]]:
    correlates = []
    intersection = roi_img1 * roi_img2
    for iid in np.unique(intersection[np.where(intersection != 0)]):
        for i1_id in np.unique(roi_img1[np.where(intersection == iid)]):
            for i2_id in np.unique(roi_img2[np.where(intersection == iid)]):
                correlates.append((i1_id, i2_id))
    return sorted(correlates)

def measure_nuc_cyto_ratio(
        client: Client, 
        project_id: int, 
        intensity_arr: np.ndarray,
        nuc_channel: str, 
        cyto_channel: str,
        measurement_channels: list[str]):

    for frame_names, labelled_arr in enumerate_rois(client, project_id):
        channels = [name.split(".")[0].split("_")[-1] for name in frame_names]
        nuc_idx = channels.index(nuc_channel)
        cyto_idx = channels.index(cyto_channel)

        whole_cell_mask = labelled_arr[cyto_idx]
        nuclear_mask = labelled_arr[nuc_idx]
        cyto_mask = whole_cell_mask * (~nuclear_mask.astype(bool)).astype(np.uint8) 

        cytoplasmic_measurements = []
        nuclear_measurements = []
        for channel in measurement_channels:
            field_intensity_arr = intensity_arr.sel(channel=channel).values

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                cytoplasmic_measurements.append({
                    "roi_id": props.label,
                    "roi_type": "cyto",
                    "channel": channel,
                    "area": props.area,
                    "integrated_intensity": props.image_intensity.sum()
                })

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                nuclear_measurements.append({
                    "roi_id": props.label,
                    "roi_type": "nuclear",
                    "channel": channel,
                    "area": props.area,
                    "integrated_intensity": props.image_intensity.sum()
                })

        nuc_df = pd.DataFrame.from_records(cytoplasmic_measurements)
        cyto_df = pd.DataFrame.from_records(nuclear_measurements)
        colocalize 
        # colocalize
    

def cli_entry(project_name: str, project_dir: pl.Path, colocalize: bool):
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

        project_id = project.id
        output_dir = project_dir / "results"
        output_dir.mkdir(exist_ok=True)
        collection = prep_experiment(project_dir, False, False, "legacy-icc", 0, None, False)
        measurement = measure_diff("GFP", "RFP", ["GFP"])
        measure_rois(client, project_id, collection, output_dir, )
        if colocalize:
            find_colocalizations(client, project_id, collection, output_dir)

