from pathlib import Path
from glob import glob
import tempfile 

from cvat_sdk import make_client, Client
from cvat_sdk.models import TaskWriteRequest
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import *
from skimage import exposure # type: ignore
import tifffile
import click
import nd2

from ..settings import settings


def get_project_id(client: Client, project_name: str) -> int | None:
    (data, _) = client.api_client.projects_api.list(
        search=project_name
    )
    if data is None or len(data.results) == 0:
        return None
    else:
        return data.results[0].id

def deploy_frames(project_name: str, paths: list[Path]):

    test = tifffile.imread(paths[0])
    convert = False
    if test.dtype != "uint8":
        convert = True
    
    with make_client(
        host=settings.cvat_url,
        credentials=(
            settings.cvat_username,
            settings.cvat_password
        )
    ) as client, tempfile.TemporaryDirectory() as tmpdir:
        
        project_id = get_project_id(client, project_name)
        if project_id is None:
            print(f"Project {project_name} does not exist; create it in the webapp first")
            return

        for path in paths:
            if convert:
                img = tifffile.imread(path)
                rescaled = exposure.rescale_intensity(img, out_range="uint8")
                outpath = Path(tmpdir) / path.name
                tifffile.imsave(outpath, rescaled)
                path = outpath
            label = path.name.replace(".tif", "")
            task_spec = TaskWriteRequest(
                name=label,
                project_id=project_id)
            try:
                client.tasks.create_from_data(
                    spec=task_spec,
                    resources=[path],
                    data_params=dict(
                        image_quality=100,
                        sorting_method="predefined")
                )
            except Exception as e:
                print(f"failed to create task for {label};")
                print(e)
                return


def deploy_ts(project_name: str, stacks: list[Path]):

    with make_client(
            host=settings.cvat_url, 
            credentials=(
                settings.cvat_username, 
                settings.cvat_password
            )
        ) as client, tempfile.TemporaryDirectory() as tmpdir:
        print(f"splitting frames in {tmpdir}")

        project_id = get_project_id(client, project_name)
        if project_id is None:
            print(f"Project {project_name} does not exist; create it in the webapp first")
            return
                
        # split into frames
        collections = {}
        for path in stacks:
            label = path.name.replace(".tif", "")
            img = tifffile.imread(path)
            collection = []
            for idx, frame in enumerate(img):
                outpath = Path(tmpdir) / f"{label}_{idx}.tif"
                rescaled = exposure.rescale_intensity(frame, out_range="uint8")
                tifffile.imsave(outpath, rescaled)
                collection.append(outpath)
            collections[label] = collection

        for label, collection in collections.items():
            task_spec = TaskWriteRequest(
                name=label,
                project_id=project_id,
            )
            try:
                client.tasks.create_from_data(
                    spec=task_spec,
                    resources=collection,
                    data_params=dict(
                        image_quality=100,
                        sorting_method="predefined")
                )
            except Exception as e:
                print(f"failed to create task for {label};")
                print(e)
                return
            
            print(f"created task for {label}")

def stage_nd2_zs(arr, label: str, tmp_path: Path, channel: str | None, apply_mip: bool):
    assert set(arr.coords) == {"Z", "Y", "X", "C"}
    if channel is not None:
        arr = arr.sel(C=channel)
    for channel, stack in arr.groupby("C"):
        stack = stack.squeeze()
        if apply_mip:
            mip = stack.max(dim="Z")
            label = f"{label}_{channel}_mip.tif"
            outpath = tmp_path / label
            arr = exposure.rescale_intensity(mip.values, out_range="uint8")
            tifffile.imsave(outpath, arr)
            yield outpath
        else:
            for idx, frame in stack.transpose("Z","Y","X"):
                label = f"{label}-{channel}-{idx}.tif"
                outpath = tmp_path / label
                arr = exposure.rescale_intensity(frame.values, out_range="uint8")
                tifffile.imsave(outpath, frame)
                yield outpath

def stage_nd2_frames(arr, label: str, tmp_path: Path):
    assert set(arr.coords) == {"Y", "X", "C"}
    label = f"{label}.tif"
    outpath = tmp_path / label
    arr = exposure.rescale_intensity(arr.values, out_range="uint8")
    tifffile.imsave(outpath, arr)
    yield outpath

def deploy_nd2s(project_name: str, paths: list[Path], channel: str | None, apply_mip: bool):
    with make_client(
            host=settings.cvat_url, 
            credentials=(
                settings.cvat_username, 
                settings.cvat_password
            )
        ) as client, tempfile.TemporaryDirectory() as tmpdir:

        tmp_path = Path(tmpdir)

        project_id = get_project_id(client, project_name)
        if project_id is None:
            print(f"Project {project_name} does not exist; create it in the webapp first")
            return

        for path in paths:
            label = path.name.replace(".nd2", "")
            arr = nd2.imread(path, xarray=True, dask=True)

            if "C" not in arr.coords:
                channel = arr.metadata["metadata"].channels[0].channel.name
                arr = arr.assign_coords(C=channel)

            if "Z" in arr.coords:
                tasks = list(stage_nd2_zs(arr, label, tmp_path, channel, apply_mip))
            else:
                tasks = list(stage_nd2_frames(arr, label, tmp_path))

            task_spec = TaskWriteRequest(
                name=label,
                project_id=project_id,
            )
            try:
                client.tasks.create_from_data(
                    spec=task_spec,
                    resources=tasks,
                    data_params=dict(image_quality=100)
                )
            except Exception as e:
                print(f"failed to create task for {label};")
                print(e)
                return
            
@click.command("cvat-deploy")
@click.argument('project_name', type=str)
@click.argument('image_glob')
@click.option('--apply-mip', is_flag=True, default=False,
    help="Applies Maximum Intensity Projection to z-stacks before uploading")
@click.option('--channel', type=str, default=None,
    help="Filters uploads to only include images from the specified channel")
def cli_entry(
    project_name: str, 
    image_glob: str, 
    apply_mip: bool, 
    channel: str | None):
    """
    Deploy images to CVAT using configured settings

    Supports single-frame images, z-stacks, and timeseries stacks.
    """

    images = [Path(path) for path in glob(image_glob)]
    nd2s = [path for path in images if path.suffix == ".nd2"]
    if len(nd2s) > 0:
        deploy_nd2s(project_name, nd2s, channel, apply_mip)
    
