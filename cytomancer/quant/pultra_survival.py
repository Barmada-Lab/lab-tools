from pathlib import Path
import logging
import warnings
import os

from PIL import Image
from sklearn.pipeline import Pipeline
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import filters, exposure, morphology, transform  # type: ignore
from dask.distributed import Worker, get_client
from stardist.models import StarDist2D, Config2D
from pystackreg import StackReg
import xarray as xr
import pandas as pd
import numpy as np

from cytomancer.utils import iter_idx_prod
from cytomancer.utils import load_experiment
from cytomancer.config import config
from cytomancer.experiment import ExperimentType, Axes
from .pultra_classifier import load_classifier

logger = logging.getLogger(__name__)


LIVE = 1
DEAD = 2

DAPI_SNR_THRESHOLD = 2


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


def predict(dapi, gfp, rfp, nuc_labels, classifier):

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


def quantify(intensity: xr.DataArray, seg_model: StarDist2D, classifier: Pipeline, annotation_dir: Path | None = None):

    def quantify_field(field: xr.DataArray):

        dapi = field.sel({Axes.CHANNEL: "DAPI"}).squeeze(drop=True).values
        gfp = field.sel({Axes.CHANNEL: "GFP"}).squeeze(drop=True).values
        rfp = field.sel({Axes.CHANNEL: "RFP"}).squeeze(drop=True).values

        footprint = morphology.disk(5)
        dapi_med = np.array([filters.median(frame, footprint) for frame in dapi])
        dapi_eqd = np.array([exposure.equalize_adapthist(frame, kernel_size=100, clip_limit=0.01) for frame in dapi_med])
        nuc_labels = np.array([seg_model.predict_instances(frame)[0] for frame in dapi_eqd]).astype(np.uint16)  # type: ignore

        preds = np.array([predict(dapi_frame, gfp_frame, rfp_frame, nuc_label_frame, classifier) for dapi_frame, gfp_frame, rfp_frame, nuc_label_frame in zip(dapi, gfp, rfp, nuc_labels)]).astype(np.uint8)
        gfp_eqd = np.array([exposure.equalize_adapthist(frame, kernel_size=100, clip_limit=0.01) for frame in gfp])
        nuc_cyto_mean = (dapi_eqd + gfp_eqd) / 2

        sr = StackReg(StackReg.RIGID_BODY)
        with warnings.catch_warnings(record=True) as w:
            tmats = sr.register_stack(nuc_cyto_mean)
            if len(w) > 0:
                # if stackreg complains, the registration is probably bad.
                # default to identity transforms
                tmats = np.array([np.eye(3) for _ in range(nuc_cyto_mean.shape[0])])

        intensity_transformed = sr.transform_stack(nuc_cyto_mean, tmats=tmats)
        labels_transformed = np.array([transform.warp(frame, tmat, order=0, mode="edge") for frame, tmat in zip(nuc_labels, tmats)])
        preds_transformed = np.array([transform.warp(frame, tmat, order=0, mode="edge") for frame, tmat in zip(preds, tmats)])
        censor_borders = np.array([transform.warp(frame, tmat, order=0, mode="constant", cval=1) for frame, tmat in zip(np.zeros_like(nuc_labels), tmats)])
        min_bb = np.max(censor_borders, axis=0).astype(bool)

        counts = []
        for label_frame, pred_frame in zip(labels_transformed, preds_transformed):
            count = 0
            for props in regionprops(label_frame):
                mask = label_frame == props.label
                if np.logical_and(mask, min_bb).sum() > 0:
                    continue

                prediction = np.argmax(np.bincount(pred_frame[mask]))
                if prediction == DEAD:
                    count += 1

            counts.append(count)

        if annotation_dir is not None:
            well_id = str(field[Axes.REGION].values[0])
            field_id = str(field[Axes.FIELD].values[0])
            output_path = annotation_dir / f"{well_id}_{field_id}.gif"

            dead = labels_transformed.copy()
            dead[np.where(preds_transformed == LIVE)] = 0

            live = labels_transformed.copy()
            live[np.where(preds_transformed == DEAD)] = 0

            # stack and invert the minimum bounding box
            censored_frames = []
            for frame in labels_transformed:
                censored_frame = frame.copy()
                censored_ids = np.unique(frame[np.where(~min_bb)])
                censored_frame[np.isin(frame, censored_ids)] = 0
                censored_frames.append(censored_frame)
            censored = np.array(censored_frames)

            intensity_img = np.stack(3 * [intensity_transformed], axis=-1)
            annotated = [mark_boundaries(frame, live_frame, color=(0, 1, 0)) for frame, live_frame in zip(intensity_img, live)]
            annotated = [mark_boundaries(frame, dead_frame, color=(1, 0, 0)) for frame, dead_frame in zip(annotated, dead)]
            annotated = [mark_boundaries(frame, c, color=(0, 0, 1)) for frame, c in zip(annotated, censored)]
            annotated = [rescale_intensity(frame, out_range="uint8") for frame in annotated]

            frame_0 = Image.fromarray(annotated[0])
            the_rest = [Image.fromarray(frame) for frame in annotated[1:]]
            frame_0.save(
                output_path, format='GIF', save_all=True,
                append_images=the_rest, duration=500, loop=0)

        return xr.DataArray(
            data=np.array(counts).reshape(-1, 1, 1),
            dims=[Axes.TIME, Axes.REGION, Axes.FIELD],
            coords={Axes.TIME: field[Axes.TIME], Axes.REGION: field[Axes.REGION], Axes.FIELD: field[Axes.FIELD]})

    chunked = intensity.chunk({
        Axes.TIME: -1,
        Axes.CHANNEL: -1,
        Axes.REGION: 1,
        Axes.FIELD: 1
    })

    # final result is cellcount as a fuction of time, region, and field
    template = chunked.isel({
        Axes.CHANNEL: 0,
        Axes.X: 0,
        Axes.Y: 0
    }).drop_vars(Axes.CHANNEL).squeeze(drop=True)

    return xr.map_blocks(quantify_field, chunked, template=template)


def run(
        experiment_path: Path,
        experiment_type: ExperimentType,
        svm_model_path: Path,
        save_annotations: bool):

    fmt = "main|%(asctime)s|%(name)s|%(levelname)s: %(message)s"
    logging.basicConfig(level=config.log_level, format=fmt)

    client = get_client()
    logger.info(f"Connected to dask scheduler {client.scheduler}")
    logger.info(f"Dask dashboard available at {client.dashboard_link}")
    logger.debug(f"Cluster: {client.cluster}")

    def init_logging(dask_worker: Worker):
        fmt = f"{dask_worker.id}|%(asctime)s|%(name)s|%(levelname)s: %(message)s"
        # disable GPU for workers. Although stardist is GPU accelerated, it's
        # faster to run many CPU workers in parallel
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.basicConfig(level=config.log_level, format=fmt)
        logging.info(dask_worker.id)

    client.register_worker_callbacks(init_logging)

    logger.debug(f"loading classifier from {svm_model_path}")
    if (classifier := load_classifier(svm_model_path)) is None:
        raise ValueError(f"Could not load classifier model at path {svm_model_path}")

    intensity = load_experiment(experiment_path, experiment_type).intensity

    if experiment_type is ExperimentType.CQ1:
        # The last row/col of DAPI images captured on the CQ1 is inexplicably zeroed out.
        # We slice it out to avoid issues further down the line
        intensity = intensity.isel({Axes.X: slice(0, 1998), Axes.Y: slice(0, 1998)})

    results_dir = experiment_path / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    if save_annotations:
        annotation_dir = results_dir / "annotations"
        annotation_dir.mkdir(exist_ok=True, parents=True)

    model = StarDist2D(Config2D(use_gpu=False)).from_pretrained("2D_versatile_fluo")
    assert model is not None, "Could not load stardist model"
    counts = quantify(intensity, model, classifier, annotation_dir=annotation_dir).compute()

    rows = []
    for ts in iter_idx_prod(counts, ignore_dims=[Axes.TIME]):
        well = ts[Axes.REGION].values
        field = ts[Axes.FIELD].values
        for time in ts[Axes.TIME]:
            count = ts.sel({Axes.TIME: time}).values
            rows.append({"well": well, "field": field, "time": time.values, "count": count})

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "survival.csv", index=False)
    logger.info(f"Finished processing {experiment_path}")
