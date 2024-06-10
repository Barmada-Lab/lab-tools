from pathlib import Path
import logging

from cvat_sdk import Client
from skimage.measure import regionprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import xarray as xr
import pandas as pd
import numpy as np
import joblib

from cytomancer.experiment import Axes
from cytomancer.cvat.helpers import new_client_from_config, get_project, get_project_label_map, enumerate_rois
from cytomancer.config import config

logger = logging.getLogger(__name__)


def build_pipeline():
    return make_pipeline(
        StandardScaler(),
        SVC(C=10, kernel="rbf", gamma="auto"))


def _transform_arrs(df, labels=None):
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
    assert df.columns.isin(["objects", "gfp", "rfp", "dapi"]).all(), \
        f"Missing columns in input X-DataFrame; expected ['objects', 'gfp', 'rfp', 'dapi'], got {df.columns.values}"

    for idx, field in df.iterrows():
        field = field.to_dict()
        objects = field["objects"]
        dapi = field["dapi"]
        gfp = field["gfp"]
        rfp = field["rfp"]

        dapi_median = np.median(dapi)
        gfp_median = np.median(gfp)
        rfp_median = np.median(rfp)

        for props in regionprops(objects):
            mask = objects == props.label
            feature_vec = {
                "dapi_signal": np.mean(dapi[mask]) / dapi_median,
                "gfp_signal": np.mean(gfp[mask]) / gfp_median,
                "rfp_signal": np.mean(rfp[mask]) / rfp_median,
                "size": mask.sum()
            }

            if labels is not None:
                is_alive = np.argmax(np.bincount(labels[idx][mask]))
                yield (feature_vec, is_alive)
            else:
                yield feature_vec


def prepare_labelled_data(df: pd.DataFrame):
    feature_vecs, labels = [], []
    for feature_vec, is_alive in _transform_arrs(df[["objects", "dapi", "gfp", "rfp"]], df["labels"]):
        feature_vecs.append(feature_vec)
        labels.append(is_alive)
    return pd.DataFrame.from_records(feature_vecs), np.array(labels)


def prepare_unlabelled_data(df: pd.DataFrame):
    return pd.DataFrame.from_records(
        _transform_arrs(df[["objects", "dapi", "gfp", "rfp"]]))


def get_segmented_image_df(client: Client, project_name: str, live_label: str, intensity: xr.DataArray):
    """
    Query CVAT, extracting segmented objects and their labels;
    attach intensity data from the provided intensity array to each field

    """

    if (project := get_project(client, project_name)) is None:
        logger.error(f"Project {project_name} not found")
        return None

    if live_label not in (label_map := get_project_label_map(client, project.id)):
        logger.error(f"Label {live_label} not found in project {project_name}; labels: {label_map}")
        return None

    live_label_id = label_map[live_label]
    records = []
    for selector, obj_arr, label_arr in enumerate_rois(client, project.id):
        subarr = intensity.sel(selector)
        gfp = subarr.sel({Axes.CHANNEL: "GFP"}).values
        rfp = subarr.sel({Axes.CHANNEL: "RFP"}).values
        dapi = subarr.sel({Axes.CHANNEL: "DAPI"}).values

        live_labels = np.zeros_like(label_arr, dtype=bool)
        live_labels[label_arr == live_label_id] = True
        annotation_frame_idx = np.argmax(np.bincount(live_labels.sum(axis=(1, 2))))

        records.append({
            "objects": obj_arr[annotation_frame_idx],
            "labels": live_labels[annotation_frame_idx],
            "gfp": gfp,
            "rfp": rfp,
            "dapi": dapi
        })

    return pd.DataFrame.from_records(records)


def train(
        project_name: str,
        live_label: str,
        intensity: xr.DataArray,
        min_dapi_snr: float | None = None) -> Pipeline | None:

    client = new_client_from_config(config)

    try:
        df = get_segmented_image_df(client, project_name, live_label, intensity)
    finally:
        client.close()

    if df is None:
        return

    X, y = prepare_labelled_data(df)

    if min_dapi_snr is not None:
        # filter low snr
        low_snr = X.index[X["dapi_signal"] < min_dapi_snr]
        X = X.drop(low_snr)
        y = np.delete(y, low_snr)

    pipe = build_pipeline()

    scores = cross_val_score(pipe, X, y, scoring="accuracy")
    logger.info(f"Fit pipeline. Cross-validation scores: {scores}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test, scoring="accuracy")  # type: ignore
    logger.info(f"Pipeline score: {score}")

    return pipe


def load_classifier(path: Path) -> Pipeline | None:
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Could not load model from {path}; {e}")
        return None
