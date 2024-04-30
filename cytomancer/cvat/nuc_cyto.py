import pathlib as pl

from cvat_sdk import Client, Config
from skimage.measure import regionprops
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import click

from cytomancer.settings import settings
from cytomancer.experiment import ExperimentType, Axes, parse_selector
from .upload import prep_experiment


def rle_to_mask(rle: list[int], width: int, height: int) -> np.ndarray:
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


def enumerate_rois(client: Client, project_id: int):
    tasks = client.projects.retrieve(project_id).get_tasks()
    for task_meta in tqdm(tasks):
        jobs = task_meta.get_jobs()
        job_id = jobs[0].id
        job_metadata, _ = client.api_client.jobs_api.retrieve_data_meta(job_id)
        frames = job_metadata.frames  # type: ignore
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
        # sometimes there are no soma corresponding to the nuclear mask
        if soma_id_contents.size == 0:
            continue
        soma_id = np.argmax(np.bincount(soma_id_contents))
        yield (nuc_id, soma_id)


# temp hack until we figure out how to work with inhomogenous experiments
def measure_nuc_cyto_ratio_nd2s(  # noqa: C901
        client: Client,
        project_id: int,
        collections: dict[str, xr.DataArray],
        nuc_channel: str,
        soma_channel: str,
        measurement_channels: list[str]):

    df = pd.DataFrame()
    for task_name, frame_names, labelled_arr in enumerate_rois(client, project_id):

        selector = parse_selector(task_name)
        region = selector.pop(Axes.REGION)
        collection = collections[region]  # type: ignore
        intensity_arr = collection.sel(selector)

        channels = selector[Axes.CHANNEL]
        nuc_idx = np.where(channels == nuc_channel)
        soma_idx = np.where(channels == soma_channel)

        soma_mask = labelled_arr[soma_idx]
        nuclear_mask = labelled_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8)

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        soma_measurements = []
        for props in regionprops(soma_mask):
            soma_measurements.append({
                "id": props.label,
                "area_soma": props.area,
            })
        soma_df = pd.DataFrame.from_records(soma_measurements)

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append({
                "id": props.label,
                "area_cyto": props.area,
            })
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append({
                "id": props.label,
                "area_nuc": props.area,
            })
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:
            # sometimes these collections are inhomogenous and don't contain all the channels we're interested in
            if channel not in intensity_arr[Axes.CHANNEL].values:
                continue

            field_intensity_arr = intensity_arr.sel({Axes.CHANNEL: channel}).values

            # measure soma
            for props in regionprops(soma_mask, intensity_image=field_intensity_arr):
                mask = soma_mask == props.label
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_mean_soma"] = field_intensity_arr[mask].mean()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_std_soma"] = field_intensity_arr[mask].std()

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                mask = cyto_mask == props.label
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_mean_cyto"] = field_intensity_arr[mask].mean()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_std_cyto"] = field_intensity_arr[mask].std()

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                mask = nuclear_mask == props.label
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_mean_nuc"] = field_intensity_arr[mask].mean()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_std_nuc"] = field_intensity_arr[mask].std()

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id").merge(soma_df, on="id")
        merged.insert(0, "label", task_name)
        df = pd.concat((df, merged))

    return df


def measure_nuc_cyto_ratio(  # noqa: C901
        client: Client,
        project_id: int,
        collection: xr.DataArray,
        nuc_channel: str,
        soma_channel: str,
        measurement_channels: list[str]):

    df = pd.DataFrame()
    for task_name, frame_names, labelled_arr in enumerate_rois(client, project_id):

        # NEW WAY, BETTER WAY, BUT DOESN'T WORK WITH OLD EXPERIMENTS :(
        selector = parse_selector(task_name)
        intensity_arr = collection.sel(selector)
        channels = selector[Axes.CHANNEL].tolist()

        # OLD WAY, BAD WAY, BUT WORKS WITH OLD EXPERIMENTS :(
        # region, field1, field2 = task_name.split("_")
        # intensity_arr = collection.sel({Axes.REGION: region, Axes.FIELD: f"{field1}_{field2}"}).load()
        # channels = [name.split(".")[0].split("_")[-1] for name in frame_names]

        nuc_idx = channels.index(nuc_channel)
        soma_idx = channels.index(soma_channel)

        soma_mask = labelled_arr[soma_idx]
        nuclear_mask = labelled_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8)

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        soma_measurements = []
        for props in regionprops(soma_mask):
            soma_measurements.append({
                "id": props.label,
                "area_soma": props.area,
            })
        soma_df = pd.DataFrame.from_records(soma_measurements)

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append({
                "id": props.label,
                "area_cyto": props.area,
            })
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append({
                "id": props.label,
                "area_nuc": props.area,
            })
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:
            # sometimes these collections are inhomogenous and don't contain all the channels we're interested in
            if channel not in intensity_arr[Axes.CHANNEL].values:
                continue

            field_intensity_arr = intensity_arr.sel({Axes.CHANNEL: channel}).values

            assert cyto_mask.shape == field_intensity_arr.shape, \
                f"cyto mask and intensity array have different shapes: {cyto_mask.shape} | {field_intensity_arr.shape}"

            # measure soma
            for props in regionprops(soma_mask, intensity_image=field_intensity_arr):
                mask = soma_mask == props.label
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_mean_soma"] = field_intensity_arr[mask].mean()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_std_soma"] = field_intensity_arr[mask].std()

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                mask = cyto_mask == props.label
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_mean_cyto"] = field_intensity_arr[mask].mean()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_std_cyto"] = field_intensity_arr[mask].std()

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                mask = nuclear_mask == props.label
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_mean_nuc"] = field_intensity_arr[mask].mean()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_std_nuc"] = field_intensity_arr[mask].std()

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id", suffixes=("_nuc", "_cyto"))
        merged.insert(0, "label", task_name)
        df = pd.concat((df, merged))

    return df


@click.command("nuc-cyto")
@click.argument("project_name", type=str)
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.argument("nuc_channel", type=str)
@click.argument("soma_channel", type=str)
@click.option("--channels", type=str, default="", help="comma-separated list of channels to measure from")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--experiment-type", type=click.Choice(ExperimentType.__members__),  # type: ignore
              callback=lambda c, p, v: getattr(ExperimentType, v) if v else None, help="experiment type")
def cli_entry(
        project_name: str,
        experiment_base: pl.Path,
        nuc_channel: str,
        soma_channel: str,
        channels: str,
        mip: bool,
        experiment_type: ExperimentType):

    client = Client(url=settings.cvat_url, config=Config(verify_ssl=False))
    client.login((settings.cvat_username, settings.cvat_password))
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
    if experiment_type is ExperimentType.ND2:
        collections = {nd2_file.name.replace(".nd2", "").replace("-", "_"): prep_experiment(nd2_file, mip, False, experiment_type, 0.0, None, False, False, False) for nd2_file in experiment_base.glob("**/*.nd2")}
        df = measure_nuc_cyto_ratio_nd2s(client, project_id, collections, nuc_channel, soma_channel, channel_list)
        df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)
    else:
        collections = prep_experiment(experiment_base, mip, False, experiment_type, 0.0, None, False, False, False)
        df = measure_nuc_cyto_ratio(client, project_id, collections, nuc_channel, soma_channel, channel_list)
        df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)
