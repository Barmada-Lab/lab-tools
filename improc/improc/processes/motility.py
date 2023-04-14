import csv

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.exposure import rescale_intensity
from skimage import filters
from skimage import measure
from skimage.morphology import disk, dilation, white_tophat

from improc.experiment.types import Channel, Dataset, Experiment, Exposure, Image, Vertex
from improc.processes.types import Task
from improc.utils import agg


def diff(arr):
    a = filters.gaussian(arr[0,0,-1], sigma=1.5)
    b = filters.gaussian(arr[0,0,0], sigma=1.5)
    footprint = disk(2)
    diff = a - dilation(b, footprint)
    diff[diff < 0] = 0
    thresh = filters.threshold_otsu(diff)
    diff[diff < thresh] = 0
    return diff

def hilight_motion(stack, window):
    """ Make sure you register before calling this function """
    sliding = sliding_window_view(stack, window_shape=(window, stack.shape[1], stack.shape[2]))
    subd = np.array(list(map(diff, sliding)))
    return subd

def calc_motility(stack, window):
    # filter noise
    l, h = np.percentile(stack, (0.5, 99.5))
    rescaled = rescale_intensity(stack, in_range=(l,h)) # type: ignore

    # handle still frame
    footprint = disk(3)
    smooth = filters.gaussian(rescaled[0], sigma=1.5)
    tophatted = white_tophat(smooth, footprint)
    still_thresh = filters.threshold_otsu(tophatted)
    threshd = np.zeros_like(tophatted)
    threshd[tophatted > still_thresh] = 1

    hilighted = hilight_motion(rescaled, window)

    labels = measure.label(hilighted.astype(bool))
    props = measure.regionprops(labels)
    avg_mito_area = np.median([prop.area for prop in props])

    hilighted_area_per_frame = np.count_nonzero(hilighted) / len(hilighted)
    pct = hilighted_area_per_frame / np.count_nonzero(threshd)

    return {
        "pct": pct,
        "counts_per_frame": hilighted_area_per_frame / avg_mito_area,
        "still": threshd,
        "motion_hilighted": hilighted
    }

class MotilityAnalysis(Task):

    def __init__(self, window: int = 10, channel: Channel = Channel.Cy5) -> None:
        super().__init__("motility")
        self.window = window
        self.channel = channel

    def group_pred(self, image: Image):
        vertex = image.get_tag(Vertex)
        exposure = image.get_tag(Exposure)
        if vertex is None or exposure is None:
            raise Exception("couldn't find vertex or exposure tags")
        return vertex, exposure

    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        groups = agg.groupby(dataset.images, self.group_pred)
        results = []
        for key, stacks in groups.items():
            _, exposure = key
            assert(len(stacks)) == 1
            if exposure.channel != self.channel:
                continue
            stack = stacks[0]
            result = calc_motility(stack, self.window)
            results.append(result)

        results_path = experiment.experiment_dir / "results"
        if not results_path.exists():
            results_path.mkdir()

        with open(results_path / "mito_moti.csv", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pct", "counts_per_frame", "still", "motion_hilighted"])
            writer.writeheader()
            writer.writerows(results)

        return dataset
