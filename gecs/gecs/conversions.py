from typing import Any

from pathlib import Path
import numpy as np

from improc.experiment import loader
from improc.experiment.types import Image, Axis, MemoryImage, Timepoint, Vertex
from improc.processes import OneToOneTask, Pipeline
from improc.processes.composite import Composite
from improc.processes.rescale import Rescale

class ConvertHack(OneToOneTask):
    def __init__(self, overwrite=False) -> None:
        super().__init__("raw_converted", overwrite=overwrite)
    
    def transform(self, image: Image) -> Image:
        # the "hack" creates multiple timepoints, even though there's really only one
        timepoint = image.get_tag(Timepoint)
        vertex = image.get_tag(Vertex)

        if timepoint is None or vertex is None:
            raise RuntimeError("WHY DOES THIS IMAGE NOT HAVE A TIMEPOINT OR VERTEX TAG?")

        vertex_label = f"ICC{timepoint.index}"
        vertex_tag = Vertex(vertex_label)

        axes = [axis for axis in image.axes if axis != Axis.T]

        tags = [vertex_tag] + [tag for tag in image.tags if not isinstance(tag, Vertex)]

        return MemoryImage(
            image.data,
            axes,
            tags
        )

def composite_icc_hack(experiment_path: Path, scratch_path: Path):
    """ 
    Composites images acquired using the imaging script "hack," 
    typically for the purposes of ICC imaging.
    """

    experiment = loader.load_experiment(experiment_path, scratch_path)
    pipeline = Pipeline(
        ConvertHack(),
        Rescale((0.5, 99.5)),
        Composite(out_depth="uint8")
    )
    pipeline.run(experiment, "raw_imgs")