from multiprocessing import Pool
from typing import Hashable, Iterable
import numpy as np
import pathlib

from improc.common.result import Result, Value

from improc.experiment import Image
from improc.experiment.types import Dataset, Experiment, Exposure, Mosaic, Vertex

from improc.processes.types import OutputCollection, Task, TaskError
from improc.utils import agg
from functools import partial
from skimage.exposure import rescale_intensity

from tqdm import tqdm

def apply_shading_correction(images: np.ndarray) -> np.ndarray:
    assert(len(images.shape) == 3)
    from pybasic import shading_correction
    basic = shading_correction.BaSiC(images)
    basic.prepare()
    basic.run()
    transformed = np.apply_over_axes(basic.normalize, images, axes=0) # type: ignore
    return transformed

class BaSiC(Task):

    def __init__(self) -> None:
        super().__init__("basic_corrected")

    def group_pred(self, image: Image) -> Hashable:
        vertex = image.get_tag(Vertex)
        mosaic_pos = image.get_tag(Mosaic)
        channel = image.get_tag(Exposure)
        return (vertex, mosaic_pos, channel)

    def correct(self, ims: list[Image]):
        arr = np.array([im.data for im in ims])
        return (ims, apply_shading_correction(arr))

    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        output = experiment.new_dataset(self.output_label)
        groups = list(agg.groupby(dataset.images, self.group_pred).values())

        with Pool() as p:
            for group, corrected in tqdm(p.imap(self.correct, groups), total=len(groups), desc=self.__class__.__name__):
                for orig, corrected_slice in zip(group, corrected):
                    tags = orig.tags
                    axes = orig.axes
                    output.write_image(corrected_slice, tags, axes)
            return Value(output)
