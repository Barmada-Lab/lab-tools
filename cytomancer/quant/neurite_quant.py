from pathlib import Path
import joblib

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.pipeline import Pipeline
import xarray as xr
import numpy as np

from cytomancer.experiment import ExperimentType, Axes
from cytomancer.oneoffs import ilastish_seg_model
from cytomancer.utils import load_experiment


def neurite_skeleseg(experiment: xr.DataArray, model: Pipeline):

    gfp = experiment.sel({Axes.CHANNEL: "GFP"})

    def segment_and_skeletonize(field):
        segmented = ilastish_seg_model.predict(field, model) == ilastish_seg_model.FOREGROUND
        skeleton = skeletonize(segmented)
        labelled = label(skeleton)
        filtered = np.zeros_like(labelled, dtype=bool)
        for props in regionprops(labelled):
            if props.area > 100:
                filtered[labelled == props.label] = True
        return filtered

    return xr.apply_ufunc(
        segment_and_skeletonize,
        gfp,
        input_core_dims=[[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool]
    ).drop_vars(Axes.CHANNEL)


def run(
        experiment_path: Path,
        experiment_type: ExperimentType,
        ilastish_model_path: Path):

    experiment = load_experiment(experiment_path, experiment_type)
    model = joblib.load(ilastish_model_path)

    skeletons = neurite_skeleseg(experiment, model)

    neurite_length = skeletons.sum(dim=[Axes.Y, Axes.X])

    results_dir = experiment_path / "results"
    results_dir.mkdir(exist_ok=True)

    neurite_length.to_dataframe("neurite_length").to_csv(results_dir / "neurite_lengths.csv")
