from pathlib import Path
import tempfile 

from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import *
import tifffile

from . settings import settings

cvat_config = Configuration(
    host=settings.cvat_url,
    username=settings.cvat_username,
    password=settings.cvat_password
)


def get_project_id(client: ApiClient, project_name: str) -> int | None:
    (data, response) = client.projects_api.list(
        search=project_name
    )
    if data is None or len(data.results) == 0:
        return None
    else:
        return data.results[0].id

def convert_to_cvat_appropriate_format(tmp_dir: Path, images: list[Path], img_dims="TYX") -> dict[str,list[Path]]:
    if img_dims == "TYX":

        # split into frames
        collections = {}
        for path in images:
            label = path.name.replace(".tif", "")
            img = tifffile.imread(path)
            collection = []
            for idx, frame in enumerate(img):
                outpath = tmp_dir / f"{label}_{idx}.tif", frame
                tifffile.imsave(outpath, frame)
                collection.append(outpath)
            collections[label] = collection
        return collections

    elif img_dims == "CYX":

        # composite into false-color image

        return {}

    elif img_dims == "CTYX":
        # composite and split
        return {}
    else:
        raise ValueError(f"Invalid or unhandled image dimensions: {img_dims}")

def create_task(client: ApiClient, project_id: int, task_name: str, image_seq: list[Path]):
    task_write_request = TaskWriteRequest(
        name=task_name,
    )

    # try:
    #     (data, response) = client.tasks_api.create(
    #     )
    # except exceptions.ApiException as e:
    #     pass

def deploy(project_name: str, img_dims: str, images: list[Path]):
    with tempfile.TemporaryDirectory() as tmpdir, ApiClient(cvat_config) as client:

        project_id = get_project_id(client, project_name)
        if project_id is None:
            print(f"Project {project_name} does not exist; create it in the webapp first")
            return
                
        #converted = convert_to_cvat_format(img_dims, images)

        # for task_name, image_seq in converted.items():
        #     create_task(client, project_id, task_name, image_seq)