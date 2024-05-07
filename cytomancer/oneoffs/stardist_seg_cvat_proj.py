from pathlib import Path

from skimage import exposure, filters, morphology  # type: ignore
from stardist.models import StarDist2D
from cvat_sdk.models import TaskAnnotationsUpdateRequest, ShapeType, LabeledShapeRequest
import numpy as np

from cytomancer.cvat.helpers import new_client_from_config, parse_selector
from cytomancer.cvat.nuc_cyto import get_rles
from cytomancer.config import config
from cytomancer.experiment import ExperimentType, Axes
from cytomancer.utils import load_experiment


def run(project_name: str, experiment_dir: Path, experiment_type: ExperimentType, channel: str, label_name: str, adapteq_clip_limit: float, median_filter_d: int, model_name: str):
    model = StarDist2D.from_pretrained(model_name)

    client = new_client_from_config(config)
    (project_data, _) = client.api_client.projects_api.list(search=project_name)

    if project_data is None or len(project_data.results) == 0:
        print(f"No projects matching query '{project_name}' found.")
        return

    try:
        project = next(filter(lambda x: x.name == project_name, project_data.results))
    except StopIteration:
        print(f"Project matching {project_name} not found.")
        return

    project_id = project.id

    intensity = load_experiment(experiment_dir, experiment_type).intensity

    (task_data, _) = client.api_client.tasks_api.list(project_id=project_id)

    if task_data is None or len(task_data.results) == 0:
        print(f"No tasks found in project '{project_name}'.")
        return

    (label_data, _) = client.api_client.labels_api.list(project_id=project_id, search=label_name)

    if label_data is None or len(label_data.results) == 0:
        print(f"No labels matching query '{label_name}' found.")
        print(f"Please create a label with the name '{label_name}' in project '{project_name}'.")
        return

    try:
        label_id = next(filter(lambda x: x.name == label_name, label_data.results)).id
    except StopIteration:
        print(f"Label matching {label_name} not found.")
        return

    for task in task_data.results:
        selector = parse_selector(task.name)
        chan_idx = int(np.where(selector[Axes.CHANNEL] == channel)[0][0])
        selector[Axes.CHANNEL] = np.array(channel)
        frame = intensity.sel(selector).values
        eqd = exposure.equalize_adapthist(frame, clip_limit=adapteq_clip_limit)
        med = filters.median(eqd, morphology.disk(median_filter_d))
        preds, _ = model.predict_instances(med)  # type: ignore

        shapes = []
        for id, rle in get_rles(preds):  # type: ignore
            shapes.append(
                LabeledShapeRequest(
                    type=ShapeType("mask"),
                    points=rle,
                    label_id=label_id,
                    frame=chan_idx,
                ))

        client.api_client.tasks_api.update_annotations(
            id=task.id,
            task_annotations_update_request=TaskAnnotationsUpdateRequest(
                shapes=shapes
            )
        )
