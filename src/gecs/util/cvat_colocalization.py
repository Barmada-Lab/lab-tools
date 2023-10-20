import itertools
import pathlib as pl

from cvat_sdk import make_client, Client
from skimage.measure import regionprops
import numpy as np
import napari
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .cvat_upload_experiment import prep_experiment
from .cvat_measure import rle_to_mask, colocalize_rois

def find_colocalizations(
        client: Client,
        project_id: int,
        arr: xr.DataArray,
        output_dir: pl.Path):
    
    tasks = client.projects.retrieve(project_id).get_tasks()
    rows = []
    for task_meta in tqdm(tasks):
        well, field1, field2 = task_meta.name.split("_")
        field = f"{field1}_{field2}"
        channel_stack = arr.sel(well=int(well), field=field)
        annotation = task_meta.get_annotations()
        height = channel_stack.y.size
        width = channel_stack.x.size

        mask = np.zeros((channel_stack.channel.size, height, width), dtype=np.uint16)
        for shape in annotation.shapes:
            frame = shape.frame
            rle = list(map(int, shape.points))
            left, top, right, bottom = rle[-4:]
            patch_height, patch_width = (bottom - top + 1, right - left +1)
            patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height).astype(np.uint16)
            mask[frame, top:bottom+1, left:right+1] = patch_mask * shape.id

        for frame_a, frame_b in itertools.combinations(mask, 2):
            for id1, id2 in colocalize_rois(frame_a, frame_b):
                rows.append({
                    "roi1": id1,
                    "roi2": id2,
                    "well": well,
                    "field": field
                })

    df = pd.DataFrame.from_records(rows)
    output_csv = output_dir / "colocalizations.csv"
    df.to_csv(output_csv, index=False)
