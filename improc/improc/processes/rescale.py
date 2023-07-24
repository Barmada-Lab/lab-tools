
import numpy as np
from skimage import exposure

from improc.experiment.types import Image, MemoryImage
from . import OneToOneTask

class Rescale(OneToOneTask):

    def __init__(self, lh: tuple[float], overwrite=False) -> None:
        super().__init__("rescaled", overwrite)
        self.lh = lh

    def transform(self, image: Image) -> Image:
        data = image.data
        l, h = np.percentile(data, self.lh)
        data = exposure.rescale_intensity(data, in_range=(l, h))
        return MemoryImage(
            data,
            image.axes,
            image.tags
        )