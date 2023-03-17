from typing import Tuple
from skimage import transform
import numpy as np

from improc.processes.types import OneToOneTask
from improc.processes.types import TaskError
from improc.experiment.types import Image, MemoryImage
from improc.common.result import Result, Value

class ScaleError(TaskError):
    ...

class Rescale(OneToOneTask):

    def __init__(self, dim: Tuple[int,int], overwrite=False) -> None:
        super().__init__("rescaled", overwrite)
        self.dim = dim

    def transform(self, image: Image) -> Result[Image, TaskError]:
        data = image.data
        rescaled = transform.resize(data, output_shape=self.dim)
        return Value(MemoryImage(rescaled, image.axes, image.tags))
