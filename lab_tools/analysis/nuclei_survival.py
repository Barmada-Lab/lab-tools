from pathlib import Path
import logging

from cvat_sdk import Client as CVATClient
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops
from joblib import dump, load
from dask.distributed import Client, Worker
import click
import xarray as xr
import pandas as pd
import numpy as np

from lab_tools.util import iter_idx_prod, experiment_type_option
from lab_tools.io import loader
from lab_tools.settings import settings
from lab_tools.experiment import ExperimentType, Axes, parse_selector
from lab_tools.cvat.nuc_cyto import rle_to_mask
from lab_tools import filters, exposure, segmentation, registration

logger = logging.getLogger(__name__)

NUCLEI_EXPERIMENT_PATH: Path = settings.collections_path / "nuclei_survival_svm_dataset"
NUCLEI_EXPERIMENT_TYPE: ExperimentType = ExperimentType.CQ1

NUCLEI_PROJECT_ID: int = 102
NUCLEI_LIVE_LABEL_ID: int = 112

SVM_MODEL_PATH = settings.models_path / "nuclei_survival_svm.joblib"

DAPI_SNR_THRESHOLD = 2

FEATURE_LABELS = [
    "dapi_signal",
    "gfp_signal",
    "rfp_signal",
    "size"
]

LIVE = 1
DEAD = 2


def get_features(mask, dapi, gfp, rfp, field_medians):
    dapi_signal = dapi[mask].mean() / field_medians[0]
    gfp_signal = gfp[mask].mean() / field_medians[1]
    rfp_signal = rfp[mask].mean() / field_medians[2]
    size = mask.astype(int).sum()
    return {
        "dapi_signal": dapi_signal,
        "gfp_signal": gfp_signal,
        "rfp_signal": rfp_signal,
        "size": size
    }


def train(nuclei_project_id: int, nuclei_live_label_id: int, nuclei_experiment: xr.DataArray, feature_f=get_features):

    client = CVATClient(settings.cvat_url)
    client.login((settings.cvat_username, settings.cvat_password))

    tasks = client.projects.retrieve(nuclei_project_id).get_tasks()

    measurements = []
    for task_meta in tasks:
        task_name = task_meta.name
        arr = nuclei_experiment.sel(parse_selector(task_name)).isel({Axes.Z: 0, Axes.TIME: 0}).load()

        dapi_field_med = np.median(arr.sel({Axes.CHANNEL: "DAPI"}).values)
        gfp_field_med = np.median(arr.sel({Axes.CHANNEL: "GFP"}).values)
        rfp_field_med = np.median(arr.sel({Axes.CHANNEL: "RFP"}).values)

        anno_table = task_meta.get_annotations()
        for shape in anno_table.shapes:
            if shape.frame != 0:
                continue
            state = shape.label_id
            rle = list(map(int, shape.points))
            l, t, r, b = rle[-4:]
            patch_height, patch_width = (b - t + 1, r - l + 1)
            patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height)

            label_arr = np.zeros((2000, 2000), dtype=bool)
            label_arr[t:b+1, l:r+1][patch_mask] = True

            measurements.append({
                "id": shape.id,
                "state": state,
                **feature_f(label_arr,
                            arr.sel({Axes.CHANNEL: "DAPI"}).values,
                            arr.sel({Axes.CHANNEL: "GFP"}).values,
                            arr.sel({Axes.CHANNEL: "RFP"}).values,
                            [dapi_field_med, gfp_field_med, rfp_field_med])
            })

    df = pd.DataFrame.from_records(measurements)
    X = df[FEATURE_LABELS]
    y = df['state'] == nuclei_live_label_id

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), SVC(C=10, gamma='auto', kernel='rbf'))
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    logger.info(f"Fit svm. Score: {score}")

    return pipe


def predict(intensity, seg, classifier):

    def _predict(dapi, gfp, rfp, nuc_labels):
        dapi_field_med = np.median(dapi)
        gfp_field_med = np.median(gfp)
        rfp_field_med = np.median(rfp)

        preds = np.zeros_like(nuc_labels)
        for props in regionprops(nuc_labels):
            mask = nuc_labels == props.label
            dapi_mean = dapi[mask].mean()

            # filter dim objects
            if dapi_mean / dapi_field_med < DAPI_SNR_THRESHOLD:
                continue

            features = get_features(mask, dapi, gfp, rfp, [dapi_field_med, gfp_field_med, rfp_field_med])
            df = pd.DataFrame.from_records([features])
            if classifier.predict(df)[0]:
                preds[mask] = LIVE
            else:
                preds[mask] = DEAD

        return preds

    return xr.apply_ufunc(
        _predict,
        intensity.sel({Axes.CHANNEL: "DAPI"}),
        intensity.sel({Axes.CHANNEL: "GFP"}),
        intensity.sel({Axes.CHANNEL: "RFP"}),
        seg,
        input_core_dims=[[Axes.Y, Axes.X], [Axes.Y, Axes.X], [Axes.Y, Axes.X], [Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int]
    )


def load_death_classifier(path: Path):
    try:
        return load(path)
    except FileNotFoundError:
        return None


def train_death_classifier(nuclei_experiment: xr.DataArray, cvat_project_id: int, nuclei_live_id: int, model_output_path: Path):
    pipe = train(cvat_project_id, nuclei_live_id, nuclei_experiment)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_output_path)
    return pipe


def annotate(intensity: xr.DataArray, classifier):
    med = filters.median(intensity, 5)
    eqd = exposure.equalize_adapthist(med, kernel_size=100, clip_limit=0.01)
    labeled = segmentation.segment_stardist(eqd)

    preds = predict(intensity, labeled.sel({Axes.CHANNEL: "DAPI"}), classifier)
    tmats = registration.register(eqd.sel({Axes.CHANNEL: "DAPI"}))
    intensity_transformed = registration.transform(eqd, tmats)
    preds_transformed = registration.transform(preds, tmats, categorical=True)
    labeled_transformed = registration.transform(labeled, tmats, categorical=True)

    return xr.Dataset({
        "intensity": intensity_transformed,
        "labels": labeled_transformed,
        "predictions": preds_transformed,
    })


def summarize(annotated: xr.Dataset, results_dir: Path, save_annotations: bool, client: Client):
    if save_annotations:
        annotation_dir = results_dir / "annotations"
        annotation_dir.mkdir(exist_ok=True, parents=True)

    for ts in iter_idx_prod(annotated, ignore_dims=[Axes.TIME, Axes.FIELD]):
        ts.load()
        return


def run(experiment_path: Path, experiment_type: ExperimentType, save_annotations: bool):

    client = Client()

    def init_logging(dask_worker: Worker):
        fmt = f"{dask_worker.id}|%(asctime)s|%(name)s|%(levelname)s: %(message)s"
        logging.basicConfig(level=settings.log_level, format=fmt, filename=f"{experiment_path.name}.log")
        logging.info(dask_worker.id)

    client.register_worker_callbacks(init_logging)

    logger.info(f"loading classifier from {SVM_MODEL_PATH}")
    if (classifier := load_death_classifier(SVM_MODEL_PATH)) is None:
        logger.info("classifier not found. Training new classifier:")
        logger.info(f"experiment path: {NUCLEI_EXPERIMENT_PATH}")
        nuclei_intensity = loader.load_experiment(NUCLEI_EXPERIMENT_PATH, NUCLEI_EXPERIMENT_TYPE).intensity
        classifier = train_death_classifier(nuclei_intensity, NUCLEI_PROJECT_ID, NUCLEI_LIVE_LABEL_ID, SVM_MODEL_PATH)

    intensity = loader.load_experiment(experiment_path, experiment_type).intensity
    annotated = annotate(intensity, classifier)
    results_dir = experiment_path / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    summarize(annotated, results_dir, save_annotations, client)


@click.command("pultra-survival")
@click.argument("experiment_path", type=click.Path(exists=True, path_type=Path))
@experiment_type_option()
@click.option("--save-annotations", is_flag=True)
def cli_entry(experiment_path: Path, experiment_type: ExperimentType, save_annotations: bool):
    run(experiment_path, experiment_type, save_annotations)
