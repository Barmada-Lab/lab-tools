from itertools import groupby
from skimage.measure import regionprops

from cvat_sdk import Client, Config
from toolz import curry
import xarray as xr
import numpy as np

from cytomancer.config import CytomancerConfig
from cytomancer.experiment import Axes


def new_client_from_config(settings: CytomancerConfig):
    client = Client(url=settings.cvat_url, config=Config(verify_ssl=False))
    client.login((settings.cvat_username, settings.cvat_password))

    org_slug = settings.cvat_org
    client.organization_slug = org_slug
    return client


# ex. field-1|region-B02|channel-GFP:RFP:Cy5|time-1:2:3:4:5:6:7:8:9:10
FIELD_DELIM = "|"
FIELD_VALUE_DELIM = "-"
VALUE_DELIM = ":"


def _fmt_coord_selector_str(label, coord_arr):
    arr = np.atleast_1d(coord_arr)
    if label == Axes.TIME:
        arr = arr.astype("long")
    if np.issubdtype(arr.dtype, np.str_):
        for value in arr:
            assert FIELD_DELIM not in value, f"{label} value {value} is invalid; contains a '|'; rename and try again"
            assert FIELD_VALUE_DELIM not in value, f"{label} value {value} is invalid; contains a '-'; rename and try again"
            assert VALUE_DELIM not in value, f"{label} value {value} is invalid; contains a ':'; rename and try again"

    return f"{label}{FIELD_VALUE_DELIM}" + VALUE_DELIM.join(map(str, arr))


@curry
def _parse_field_selector(arr: xr.DataArray | None, selector: str):
    field_name, field_values = selector.split(FIELD_VALUE_DELIM)
    try:
        axis = Axes(field_name)
    except ValueError:
        try:
            axis = Axes(field_name.split(".")[-1].lower())
        except ValueError:
            raise ValueError(f"Invalid field name {field_name} in selector {selector}")

    if arr is None:
        target_dtype = np.str_
    else:
        target_dtype = arr[axis].dtype

    match axis:
        case Axes.TIME:
            field_value_tokens = np.array([np.datetime64(int(ts), 'ns')for ts in field_values.split(VALUE_DELIM)])
            if field_value_tokens.size == 1:
                field_value = field_value_tokens[0]
                return (axis, field_value)
            else:
                return (axis, field_value_tokens)
        case _:
            field_value_tokens = np.array(field_values.split(VALUE_DELIM)).astype(target_dtype)
            if field_value_tokens.size == 1:
                field_value = field_value_tokens[0]
                return (axis, field_value)
            else:
                return (axis, field_value_tokens)


def coord_selector(arr: xr.DataArray) -> str:
    """Derives a string-formatted selector from an array's coordinates."""
    coords = sorted(arr.coords.items())
    filtered = filter(lambda coord: coord[0] not in [Axes.X, Axes.Y], coords)
    return FIELD_DELIM.join([
        _fmt_coord_selector_str(axis.value, coord.values) for axis, coord in filtered  # type: ignore
    ])


def parse_selector(selector_str: str) -> dict[Axes, np.ndarray]:
    """Parses a selector string into a dictionary of axes to values"""
    return dict(map(_parse_field_selector(None), selector_str.split(FIELD_DELIM)))  # type: ignore


def rle_to_mask(rle: list[int], width: int, height: int) -> np.ndarray:
    assert sum(rle) == width * height, "RLE does not match image size"

    decoded = [0] * (width * height)  # create bitmap container
    decoded_idx = 0
    value = 0

    for v in rle:
        decoded[decoded_idx:decoded_idx + v] = [value] * v
        decoded_idx += v
        value = abs(value - 1)

    decoded = np.array(decoded, dtype=bool)
    decoded = decoded.reshape((height, width))  # reshape to image size
    return decoded


def mask_to_rle(mask: np.ndarray) -> list[int]:
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.flatten())):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return counts


def get_labelled_arr(anno_table, length, height, width):
    channel_stack_mask = np.zeros((length, height, width), dtype=int)
    for shape in anno_table.shapes:
        id = shape.id
        frame = shape.frame
        rle = list(map(int, shape.points))
        left, top, right, bottom = rle[-4:]
        patch_height, patch_width = (bottom - top + 1, right - left + 1)
        patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height)
        channel_stack_mask[frame, top:bottom + 1, left:right + 1][patch_mask] = id
    return channel_stack_mask


def get_rles(labelled_arr: np.ndarray):
    rles = []
    for props in regionprops(labelled_arr):
        id = props.label
        mask = labelled_arr == id
        top, left, bottom, right = props.bbox
        rle = mask_to_rle(mask[top:bottom, left:right])
        rle += [left, top, right-1, bottom-1]

        left, top, right, bottom = rle[-4:]
        patch_height, patch_width = (bottom - top + 1, right - left + 1)
        patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height)

        assert np.all(patch_mask == mask[top:bottom+1, left:right+1])
        rles.append((id, rle))
    return rles


def enumerate_rois(client: Client, project_id: int):
    tasks = client.projects.retrieve(project_id).get_tasks()
    tasks = client.projects.retrieve(project_id).get_tasks()
    for task_meta in tasks:
        jobs = task_meta.get_jobs()
        job_id = jobs[0].id
        job_metadata, _ = client.api_client.jobs_api.retrieve_data_meta(job_id)
        frames = job_metadata.frames  # type: ignore
        width = frames[0].width
        height = frames[0].height
        anno_table = task_meta.get_annotations()
        labelled_arr = get_labelled_arr(anno_table, len(frames), height, width)
        task_name = task_meta.name
        selector = parse_selector(task_name)
        yield selector, labelled_arr


def get_project(client: Client, project_name: str):
    """
    Returns a project with the given name, or None if no such project exists.
    """
    for project in client.projects.list():
        if project.name == project_name:
            return project
    return None


def load_project_segmentation(client: Client, project_id: int, intensity: xr.DataArray):
    """
    Creates a DataArray containing all the masks for a given project, and creates an xarray Dataset
    containing the original intensity array and the generated mask array.
    """
    labels = xr.zeros_like(intensity).load()
    for selector, labelled_arr in enumerate_rois(client, project_id):
        labels.loc[selector] = labelled_arr
    return xr.Dataset({
        "intensity": intensity,
        "labels": labels
    })
