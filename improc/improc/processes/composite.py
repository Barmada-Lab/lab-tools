from typing import Hashable

from skimage import exposure #type: ignore
import numpy as np

from improc.experiment.types import Image, Vertex, Exposure, Mosaic, Timepoint, Channel, MemoryImage
from . import ManyToOneTask

class Composite(ManyToOneTask):

    channel_colors = {
        Channel.DAPI: "#007fff",
        Channel.RFP: "#ffe600",
        Channel.GFP: "#00ff00",
        Channel.Cy5: "#ff0000",
        Channel.White: "#ffffff",
        Channel.BRIGHTFIELD: "#ffffff",
    }

    def __init__(self, out_depth="uint16", overwrite=False) -> None:
        super().__init__("composited", overwrite)
        self.out_depth = out_depth

    def group_pred(self, image: Image) -> Hashable:
        return (
            image.get_tag(Vertex),
            image.get_tag(Mosaic),
            image.get_tag(Timepoint)
        )

    def color_img(self, data: np.ndarray, channel: Channel):
        data = exposure.rescale_intensity(data, out_range=np.float32)
        data /= data.max()
        data_rgb = np.stack([data, data, data], axis=-1)
        h = self.channel_colors[channel][1:]
        r, g, b = [int(h[i:i+2], 16) for i in (0,2,4)]
        color_map = np.stack(
            (np.ones_like(data) * r, np.ones_like(data) * g, np.ones_like(data) * b),
            axis=-1,
            dtype=np.float32
        )
        color_map /= 255.0
        data_rgb *= color_map
        return data_rgb

    def transform(self, images: list[Image]) -> Image:
        ordered_images = sorted(images, key=lambda image: image.get_tag(Exposure).channel) # type: ignore

        data = np.sum([self.color_img(image.data, image.get_tag(Exposure).channel) for image in ordered_images], axis=0)
        rescaled = exposure.rescale_intensity(data, out_range=self.out_depth)
        axes = ordered_images[0].axes
        tags = [tag for tag in images[0].tags if not isinstance(tag, Exposure)]
        return MemoryImage(rescaled, axes, tags)
