from typing import Callable, Hashable

import numpy as np
from skimage import morphology
from pystackreg import StackReg
from skimage.filters import sobel
import largestinteriorrectangle as lir

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

    def transform(self, images: list[Image]) -> Image:
        ordered = np.array([img.data for img in sorted(images, key=lambda x: x.get_tag(Timepoint).index)]) # type: ignore
        tags = list(filter(lambda x: not isinstance(x, Timepoint), images[0].tags))
        axes = [Axis.T] + images[0].axes
        sr = StackReg(StackReg.RIGID_BODY)
        reg_stack = np.array([self.registration_transform(img) for img in ordered])

        time_axis = sr._detect_time_axis(reg_stack)
        if time_axis != 0 and not self.force_bad_reg: # If the registration is gonna be garbage, don't bother
            print(f"Bad registration for {images[0].vertex}; defaulting to stack w/o registration")
            return MemoryImage(ordered, axes, tags)

        transforms = sr.register_stack(reg_stack, reference="previous")
        stacked = sr.transform_stack(np.array(ordered), tmats=transforms)
        if self.crop_output:
            try:
                stacked = crop(stacked)
            except:
                raise Exception(f"Can't crop image: {images[0]}")
        return MemoryImage(stacked, axes, tags)

def crop(stack: np.ndarray) -> np.ndarray:

    min_poly = np.prod(stack != 0, axis=0)
    x1,y1,x2,y2 = lir.lir(min_poly.astype(bool))

    return stack[:, y1:y2, x1:x2]

def composite_stack(stacks: np.ndarray, tmats: np.ndarray) -> np.ndarray | None:
    assert stacks.ndim == 4

    sr = StackReg(StackReg.RIGID_BODY)
    stacked_cropped = np.array([crop(sr.transform_stack(stack, tmats=tmats)) for stack in stacks])
    return stacked_cropped
