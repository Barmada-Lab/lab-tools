
from typing import Hashable
from improc.common.result import Result, Value
from improc.experiment.types import Axis, Exposure, Image, MemoryImage, Timepoint, Vertex
from improc.processes.types import ManyToOneTask, TaskError

from pystackreg import StackReg
import numpy as np

class Stack(ManyToOneTask):

    def __init__(self, crop_output: bool = True) -> None:
        super().__init__("stacked")
        self.crop_output = crop_output

    def group_pred(self, image: Image) -> Hashable:
        return (image.get_tag(Vertex), image.get_tag(Exposure))

    def crop(self, stack: np.ndarray) -> np.ndarray:
        # offset from image edges
        max_left_offset = 0
        max_right_offset = 0
        max_top_offset = 0
        max_bot_offset = 0
        for frame in stack:
            # find rows/columns that are entirely zero
            col_borders = (frame == 0).all(axis=0)
            row_borders = (frame == 0).all(axis=1)
            first_false = lambda x: np.where(x == False)[0][0]

            if (left_offset := first_false(col_borders)) > max_left_offset:
                max_left_offset = left_offset

            if (right_offset := first_false(col_borders[::-1])) > max_right_offset:
                max_right_offset = right_offset

            if (top_offset := first_false(row_borders)) > max_top_offset:
                max_top_offset = top_offset

            if (bot_offset := first_false(row_borders)) > max_bot_offset:
                max_bot_offset = bot_offset

        # convert offsets back to array indices
        last_row = stack.shape[2] - 1
        last_col = stack.shape[1] - 1

        min_row_idx = max_top_offset
        max_row_idx = last_row - max_bot_offset
        min_col_idx = max_left_offset
        max_col_idx = last_col - max_right_offset

        return stack[:, min_row_idx:max_row_idx, min_col_idx:max_col_idx]

    def transform(self, images: list[Image]) -> Result[Image, TaskError]:
        ordered = [img.data for img in sorted(images, key=lambda x: x.get_tag(Timepoint).index)] # type: ignore
        sr = StackReg(StackReg.RIGID_BODY)
        stacked = sr.register_transform_stack(np.array(ordered), reference="previous")
        if self.crop_output:
            stacked = self.crop(stacked)
        tags = list(filter(lambda x: not isinstance(x, Timepoint), images[0].tags))
        axes = [Axis.T] + images[0].axes
        return Value(MemoryImage(stacked, axes, tags))
