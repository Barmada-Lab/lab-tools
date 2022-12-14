from typing import Callable, Hashable

import numpy as np
from pystackreg import StackReg
from skimage.filters import sobel

from improc.common.result import Error, Result, Value
from improc.experiment.types import Axis, Exposure, Image, MemoryImage, Timepoint, Vertex
from improc.processes.types import ManyToOneTask, TaskError

class BadImageCantCrop(TaskError):
    ...

class Stack(ManyToOneTask):

    def __init__(self, registration_transform: Callable[[np.ndarray], np.ndarray] = sobel,  crop_output: bool = True, force_bad_reg: bool = False) -> None: # type: ignore
        super().__init__("stacked")
        self.crop_output = crop_output
        self.registration_transform = registration_transform
        self.force_bad_reg = force_bad_reg

    def group_pred(self, image: Image) -> Hashable:
        return (image.get_tag(Vertex), image.get_tag(Exposure))

    def transform(self, images: list[Image]) -> Result[Image, TaskError]:
        ordered = np.array([img.data for img in sorted(images, key=lambda x: x.get_tag(Timepoint).index)]) # type: ignore
        tags = list(filter(lambda x: not isinstance(x, Timepoint), images[0].tags))
        axes = [Axis.T] + images[0].axes
        sr = StackReg(StackReg.RIGID_BODY)
        reg_stack = np.array([self.registration_transform(img) for img in ordered])

        time_axis = sr._detect_time_axis(reg_stack)
        if time_axis != 0 and not self.force_bad_reg: # If the registration is gonna be garbage, don't bother
            print(f"Bad registration for {images[0].vertex}; defaulting to stack w/o registration")
            return Value(MemoryImage(ordered, axes, tags))

        transforms = sr.register_stack(reg_stack, reference="previous")
        stacked = sr.transform_stack(np.array(ordered), tmats=transforms)
        if self.crop_output:
            try:
                stacked = crop(stacked)
            except:
                print(images[0])
                return Error(BadImageCantCrop())
        return Value(MemoryImage(stacked, axes, tags))

def crop(stack: np.ndarray) -> np.ndarray:
    # offset from image edges
    max_left_offset = 0
    max_right_offset = 0
    max_top_offset = 0
    max_bot_offset = 0

    def first_false(x):
        query = np.where(x == False)
        if query[0].any():
            return query[0][0]
        else:
            return 0

    for frame in stack:
        # find rows/columns that are entirely zero
        col_borders = (frame == 0).all(axis=0)
        row_borders = (frame == 0).all(axis=1)

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
