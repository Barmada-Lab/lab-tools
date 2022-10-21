from collections.abc import Iterable
from skimage.morphology import disk, white_tophat
from skimage.exposure import exposure, rescale_intensity

import numpy as np
from improc.common.result import Error, Result, Value
from improc.experiment.types import Axis, Channel, Dataset, Experiment, Exposure, Image, MemoryImage

from improc.processes.types import OneToOneTask, Task, TaskError, UnhandledShape

def ineuron_preprocess(image: np.ndarray, r_disk: int) -> np.ndarray:

    dl, dh = np.percentile(image, (0.5, 99.5))
    rescaled = rescale_intensity(image, in_range=(dl, dh)) # type: ignore
    footprint = disk(r_disk)
    tophatted = white_tophat(rescaled, footprint)
    return tophatted # type: ignore


class Brightfield_iNeuron_Preprocess(OneToOneTask):
    """
    Brightfield preprocessing task tuned for iNeurons.

    Enhances contrast while maintaining legibility of features.

    Expects 2d single-channel images.

    Parameters
    ----------

    r_disk : int
        radius of disk footprint to use for tophat transform

    """

    def __init__(self, r_disk: int = 8) -> None:
        super().__init__("brightfield_preprocessed")
        self.r_disk = r_disk

    def filter(self, images: Iterable[Image]) -> Iterable[Image]:
        for image in images:
            if (exposure := image.get_tag(Exposure)) is not None and exposure.channel == Channel.BRIGHTFIELD:
                yield image

    def transform(self, image: Image) -> Result[Image, TaskError]:
        preprocessed = ineuron_preprocess(image.data, self.r_disk)
        return Value(MemoryImage(preprocessed, image.axes, image.tags))
