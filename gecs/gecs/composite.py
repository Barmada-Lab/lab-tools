from typing import Any

from pathlib import Path
from collections import defaultdict
import tifffile
from skimage import exposure, filters # type: ignore
from pystackreg import StackReg
import numpy as np

from survival.gedi import _preprocess_gedi_rfp
from improc.experiment import loader, Experiment
from improc.experiment.types import Image, Axis, MemoryImage, Timepoint, Vertex, Channel, Exposure, Mosaic
from improc.processes import OneToOneTask, Pipeline
from improc.processes import BaSiC
from improc.processes import stack
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

def composite_icc_hack(experiment: Experiment):
    """ 
    Composites images acquired using the imaging script "hack," 
    typically for the purposes of ICC imaging.
    """

    pipeline = Pipeline(
        ConvertHack(),
        Composite(out_depth="uint8")
    )
    pipeline.run(experiment, "raw_imgs")

def composite_survival(experiment: Experiment, ignore_channels: list[Channel] = []):
    """ 
    Composites images acquired using the imaging script "survival," 
    typically for the purposes of survival imaging.
    """

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

    if not (experiment.scratch_dir / "composited").exists():
        (experiment.scratch_dir / "composited").mkdir()

    def ordering_key(img):
        return img.get_tag(Timepoint).index

    for (vertex, mosaic), images in timeseries.items():

        chans = defaultdict(list)
        for image in images:
            chan = image.get_tag(Exposure).channel
            chans[chan].append(image)

        gfp = [im.data for im in sorted(chans[Channel.GFP], key=ordering_key)]
        gfp_norm = _preprocess_gedi_rfp(np.array(gfp))
        # gfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.2) for frame in gfp_norm])
        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(gfp_norm)
        gfp_registered = sr.transform_stack(gfp_norm, tmats=tmats)
        gfp_registered = stack.crop(gfp_registered)

        reg_timeseries = { Channel.GFP: gfp_registered }
        for chan, images in chans.items():
            if chan == Channel.GFP or chan in ignore_channels:
                continue

            frames = [im.data for im in sorted(images, key=ordering_key)]
            frames_norm = _preprocess_gedi_rfp(np.array(frames))
            # frames_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.2) for frame in frames_norm])
            frames_reg = sr.transform_stack(frames_norm, tmats=tmats)
            frames_reg = stack.crop(frames_reg)
            reg_timeseries[chan] = frames_reg

        sum_img = np.zeros((*gfp_registered.shape,3))
        composite = Composite(out_depth="uint32")
        for chan, frames in reg_timeseries.items():
            colored = np.array([composite.color_img(frame, chan) for frame in frames])
            sum_img += colored # type: ignore
        
        sum_img = exposure.rescale_intensity(sum_img, out_range="uint8")
        mosaic_label = "_".join(map(str, mosaic.index))
        tifffile.imwrite(experiment.scratch_dir / "composited" / f"{vertex.label}-{mosaic_label}.tif", sum_img)

def cli_entry(args):
    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(args.experiment_dir, scratch_dir)
    if args.icc_hack:
        composite_icc_hack(experiment)
    else:
        composite_survival(experiment, args.ignore)