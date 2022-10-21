
from typing import Hashable
from improc.common.result import Result, Value
from improc.experiment.types import Axis, Exposure, Image, MemoryImage, Timepoint, Vertex, get_tag
from improc.processes.types import ManyToOneTask, Task, TaskError

from pystackreg import StackReg
import numpy as np

class Stack(ManyToOneTask):

    def __init__(self) -> None:
        super().__init__("stacked")

    def group_pred(self, image: Image) -> Hashable:
        return (image.get_tag(Vertex), image.get_tag(Exposure))

    def transform(self, images: list[Image]) -> Result[Image, TaskError]:
        ordered = [img.data for img in sorted(images, key=lambda x: x.get_tag(Timepoint).index)] # type: ignore
        sr = StackReg(StackReg.RIGID_BODY)
        stacked = sr.register_transform_stack(np.array(ordered), reference="previous")
        tags = list(filter(lambda x: not isinstance(x, Timepoint), images[0].tags))
        axes = [Axis.T] + images[0].axes
        return Value(MemoryImage(stacked, axes, tags))
