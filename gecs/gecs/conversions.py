from typing import Any

from pathlib import Path
from skimage import filters
from collections import defaultdict
import tifffile
from skimage import exposure
from pystackreg import StackReg
import numpy as np

from survival.gedi import _preprocess_gedi_rfp
from improc.experiment import loader
from improc.experiment.types import Image, Axis, MemoryImage, Timepoint, Vertex, Channel, Exposure, Mosaic
from improc.processes import OneToOneTask, Pipeline
from improc.processes import BaSiC
from improc.processes.composite import Composite

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

def reg_preprocessing(img: np.ndarray) -> np.ndarray:
    norm = _preprocess_gedi_rfp(np.array((img, img)))[0] # just normalizes
    filtered = filters.butterworth(norm, high_pass=False, cutoff_frequency_ratio=0.1)
    return filtered

def composite_survival(experiment_path, scratch_path: Path):
    """ 
    Composites images acquired using the imaging script "survival," 
    typically for the purposes of survival imaging.
    """

    experiment = loader.load_experiment(experiment_path, scratch_path)
    if not "basic_corrected" in experiment.datasets:
        pipeline = Pipeline(
            BaSiC()
        )
        pipeline.run(experiment, "raw_imgs")

    timeseries = defaultdict(list)
    for image in experiment.datasets["basic_corrected"].images:
        vertex = image.get_tag(Vertex)
        mosaic = image.get_tag(Mosaic)
        timeseries[(vertex, mosaic)].append(image)

    if not (scratch_path / "composited").exists():
        (scratch_path / "composited").mkdir()

    def ordering_key(img):
        return img.get_tag(Timepoint).index

    for (vertex, mosaic), images in timeseries.items():

        chans = defaultdict(list)
        for image in images:
            chan = image.get_tag(Exposure).channel
            chans[chan].append(image)

        gfp = [im.data for im in sorted(chans[Channel.GFP], key=ordering_key)]
        gfp_norm = _preprocess_gedi_rfp(np.array(gfp))
        gfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.1) for frame in gfp_norm])
        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(gfp_filtered)
        gfp_registered = sr.transform_stack(gfp_filtered, tmats=tmats)

        reg_timeseries = { Channel.GFP: gfp_registered }
        for chan, images in chans.items():
            if chan == Channel.GFP or chan == Channel.BRIGHTFIELD or chan == Channel.White:
                continue

            frames = [im.data for im in sorted(images, key=ordering_key)]
            frames_norm = _preprocess_gedi_rfp(np.array(frames))
            frames_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.1) for frame in frames_norm])
            frames_reg = sr.transform_stack(frames_filtered, tmats=tmats)
            reg_timeseries[chan] = frames_reg

        sum_img = np.zeros((*gfp_registered.shape,3))
        composite = Composite(out_depth="uint32")
        for chan, frames in reg_timeseries.items():
            colored = np.array([composite.color_img(frame, chan) for frame in frames])
            sum_img += colored
        
        sum_img = exposure.rescale_intensity(sum_img, out_range="uint8")
        mosaic_label = "_".join(mosaic.index)
        tifffile.imwrite(scratch_path / "composited" / f"{vertex.label}-{mosaic_label}.tif", sum_img)