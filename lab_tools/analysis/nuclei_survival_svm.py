from itertools import product
from pathlib import Path
import pathlib as pl

from cvat_sdk import Client
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage import filters, morphology, exposure, segmentation  # type: ignore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops
from joblib import dump, load
from stardist.models import StarDist2D
from tqdm import tqdm
from PIL import Image
import xarray as xr
import pandas as pd
import numpy as np
import click

from lab_tools.io import loader
from lab_tools.settings import settings
from lab_tools.experiment import Axes, parse_selector, ExperimentType
from lab_tools.cvat.nuc_cyto import rle_to_mask


TURBO_SHARED = Path("/nfs/turbo/shared")
SVM_MODEL_PATH: Path = TURBO_SHARED / "models" / "nuclei_survival_svm.joblib"
SVM_DATA_DIR: Path = TURBO_SHARED / "collections" / "nuclei_survival_svm_dataset"
NUCLEI_PROJECT_ID: int = 102
NUCLEI_EXPERIMENT_TYPE: ExperimentType = ExperimentType.CQ1
NUCLEI_LIVE_LABEL_ID: int = 112

DAPI_SNR_THRESHOLD = 2

FEATURE_LABELS = [
    "dapi_signal",
    "gfp_signal",
    "rfp_signal",
    "size"
]

LIVE = 1
DEAD = 2


def get_features(arr, mask, field_medians):
    dapi = arr.sel({Axes.CHANNEL: "DAPI"}).values[mask].mean()
    gfp = arr.sel({Axes.CHANNEL: "GFP"}).values[mask].mean()
    rfp = arr.sel({Axes.CHANNEL: "RFP"}).values[mask].mean()
    size = mask.astype(int).sum()
    return {
        "dapi_signal": dapi / field_medians[0],
        "gfp_signal": gfp / field_medians[1],
        "rfp_signal": rfp / field_medians[2],
        "size": size
    }


def train(nuclei_project_id: int, nuclei_live_label_id: int, nuclei_experiment: xr.Dataset, feature_f=get_features):

    client = Client(settings.cvat_url)
    client.login((settings.cvat_username, settings.cvat_password))

    tasks = client.projects.retrieve(nuclei_project_id).get_tasks()

    measurements = []
    for task_meta in tasks:
        task_name = task_meta.name
        arr = nuclei_experiment.intensity.sel(parse_selector(task_name)).isel({Axes.Z: 0, Axes.TIME: 0}).load()

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
                **feature_f(arr, label_arr, [dapi_field_med, gfp_field_med, rfp_field_med])
            })

    df = pd.DataFrame.from_records(measurements)
    X = df[FEATURE_LABELS]
    y = df['state'] == nuclei_live_label_id

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), SVC(C=10, gamma='auto', kernel='rbf'))
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"Fit svm. Score: {score}")

    return pipe


def evaluate(arr, seg_model, pipe):

    dapi_field_med = np.median(arr.sel({Axes.CHANNEL: "DAPI"}).values)
    gfp_field_med = np.median(arr.sel({Axes.CHANNEL: "GFP"}).values)
    rfp_field_med = np.median(arr.sel({Axes.CHANNEL: "RFP"}).values)

    dapi = arr.sel({Axes.CHANNEL: "DAPI"}).values

    med_filt = filters.median(dapi, morphology.disk(5))
    clahed = exposure.equalize_adapthist(med_filt, kernel_size=100, clip_limit=0.01)
    objects, _ = seg_model.predict_instances(clahed)
    preds = np.zeros_like(objects, dtype=int)

    for props in regionprops(objects):
        mask = objects == props.label
        dapi_mean = dapi[mask].mean()

        # filter dim objects
        if dapi_mean / dapi_field_med < DAPI_SNR_THRESHOLD:
            objects[mask] = 0
            continue

        features = pd.DataFrame.from_records([get_features(arr, mask, [dapi_field_med, gfp_field_med, rfp_field_med])])
        if pipe.predict(features)[0]:
            preds[mask] = LIVE
        else:
            preds[mask] = DEAD

    return objects, preds


def summarize(raw, pipe, gif_path=None):

    seg_model = StarDist2D.from_pretrained("2D_versatile_fluo")

    count_rows = []
    for well, field in tqdm(product(raw.coords[Axes.REGION], raw.coords[Axes.FIELD])):
        stack = raw.sel({
            Axes.REGION: well.values,
            Axes.FIELD: field.values,
        }).isel({Axes.X: slice(0, 1998), Axes.Y: slice(0, 1998)}).load()
        frames = []
        annotated = []
        for t in raw.coords[Axes.TIME]:

            data = stack.sel({Axes.TIME: t})

            objects, preds = evaluate(data, seg_model, pipe)

            if len(frames) == 0:
                frames.append(preds == DEAD)

            n_alive = 0
            for props in regionprops(objects):
                mask = objects == props.label
                size = props.area
                overlap = frames[-1][mask].astype(int).sum()
                if overlap / size > 0.5:
                    # this cell previously died maybe
                    preds[mask] = DEAD  # persist dead to this timepoint to prevent zombies
                    continue

                if (preds[mask] == LIVE).all():
                    n_alive += 1

            frames.append(preds == DEAD)

            count_rows.append({
                "well": str(well.values),
                "field": str(field.values),
                "time": t.values,
                "count": n_alive
            })

            gfp = data.sel({Axes.CHANNEL: "GFP"}).values
            dead = preds == DEAD
            rescaled = exposure.rescale_intensity(gfp, out_range='uint8')
            marked = segmentation.mark_boundaries(
                rescaled, dead, color=(1, 0, 0), mode="thick")
            marked = segmentation.mark_boundaries(
                marked, preds == LIVE, color=(0, 1, 0), mode="thick")
            marked = exposure.rescale_intensity(marked, out_range='uint8')
            annotated.append(marked)

        if gif_path is not None:
            if not gif_path.exists():
                gif_path.mkdir(parents=True)

            path = gif_path / f"{well.values}_{field.values}.gif"
            data = np.array(annotated)
            frame_0 = Image.fromarray(data[0])
            the_rest = [Image.fromarray(frame) for frame in data[1:]]
            frame_0.save(
                path, format='GIF', save_all=True,
                append_images=the_rest, duration=500, loop=0)

    df = pd.DataFrame.from_records(count_rows)
    return df


def run(experiment: xr.Dataset, output_dir: pl.Path):
    if not SVM_MODEL_PATH.exists():
        nuclei_experiment = loader.load_experiment(SVM_DATA_DIR, NUCLEI_EXPERIMENT_TYPE)
        pipe = train(NUCLEI_PROJECT_ID, NUCLEI_LIVE_LABEL_ID, nuclei_experiment)
        SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(pipe, SVM_MODEL_PATH)
    else:
        pipe = load(SVM_MODEL_PATH)

    gif_path = output_dir / "annotated"
    gif_path.mkdir(parents=True, exist_ok=True)

    data = experiment.intensity.isel({Axes.Z: 0})
    summary = summarize(data, pipe, gif_path)
    summary.to_csv(output_dir / "nuclei_survival.csv", index=False)


@click.command("nuclei-survival")
@click.argument("experiment_base", type=click.Path(path_type=pl.Path))
@click.argument("output_dir", type=click.Path(path_type=pl.Path))
def cli_entry(experiment_base: pl.Path, output_dir: pl.Path):
    experiment = loader.load_experiment(experiment_base, ExperimentType.CQ1)
    output_dir.mkdir(parents=True, exist_ok=True)
    run(experiment, output_dir)
