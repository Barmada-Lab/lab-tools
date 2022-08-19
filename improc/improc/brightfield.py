from math import ceil
from typing import Generator, Iterator
from skimage.morphology import disk, white_tophat
from skimage.exposure import rescale_intensity

from common.experiments.image import Image
from common.experiments import tags

import numpy as np
import random


def ineuron_preprocess(image: np.ndarray, r_disk=8) -> np.ndarray:
    """
    Preprocesses white light images of ineurons.

    Parameters
    ----------

    images : list[Image]
        iterable of 2d single-channel images

    sample_k : int
        sample size for normalization

    footprint_r : int
        radius of disk footprint to use for tophat transform

    """

    dl, dh = np.percentile(image, (0.5, 99.5))
    rescaled = rescale_intensity(image, in_range=(dl, dh))
    footprint = disk(r_disk)
    tophatted = white_tophat(rescaled, footprint)
    return tophatted
