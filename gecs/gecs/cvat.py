from pathlib import Path
import tempfile 

from skimage import exposure
from cvat_sdk import make_client, Client
from cvat_sdk.models import TaskWriteRequest
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import *
import tifffile

from . settings import settings


def get_project_id(client: Client, project_name: str) -> int | None:
    (data, response) = client.api_client.projects_api.list(
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
                project_id=project_id
            )
            try:
                client.tasks.create_from_data(
                    spec=task_spec,
                    resources=[path]
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
                    resources=collection
                )
            except Exception as e:
                print(f"failed to create task for {label};")
                print(e)
                return
            
            print(f"created task for {label}")

def cli_entry(args):
    if args.ts:
        deploy_ts(args.project_name, args.images)
    else:
        deploy_frames(args.project_name, args.images)