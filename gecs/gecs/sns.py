from pathlib import Path
from collections import defaultdict

from skimage import exposure, filters
from pystackreg import StackReg
import tifffile
import numpy as np
import os

from survival.gedi import _preprocess_gedi_rfp
from improc.experiment.types import Vertex, Mosaic, Exposure, Channel, Timepoint
from improc.experiment import loader
from improc.processes import BaSiC, Stitch, Pipeline, Task
from improc.processes.stack import crop

def format_path(scratch_path: Path, legacy: bool, vertex: Vertex, mosaic: Mosaic | None, channel: Channel):
    if legacy:
        base = scratch_path / "stacked" / channel
    else:
        base= scratch_path / "stacked"
    
    if mosaic is None:
        return base / f"{vertex.label}-{channel}.tif"
    else:
        return base / f"{vertex.label}-{mosaic.index[0]}_{mosaic.index[1]}-{channel}.tif"

def stitch_n_stack(experiment_path: Path, scratch_path: Path, legacy: bool, out_range: str = "uint16", stitch: bool = True):

    experiment = loader.load_experiment(experiment_path, scratch_path)
    steps: list[Task] = [ BaSiC() ]
    if stitch:
        steps.append(Stitch())
    initial_pipeline = Pipeline(*steps)
    initial_pipeline.run(experiment, "raw_imgs")

    # now stack based on GFP registration
    locs = defaultdict(list)
    if stitch:
        for image in experiment.datasets["stitched"].images:
            vertex = image.get_tag(Vertex)
            locs[(vertex, None)].append(image)
    else:
        for image in experiment.datasets["basic_corrected"].images:
            vertex = image.get_tag(Vertex)
            mosaic = image.get_tag(Mosaic)
            locs[(vertex, mosaic)].append(image)

    def sorting_key(image):
        return image.get_tag(Timepoint).index

    for (vertex, mosaic), images in locs.items():
        chans = defaultdict(list)
        for image in images:
            chan = image.get_tag(Exposure).channel
            chans[chan].append(image)
        
        gfp_imgs = sorted(chans[Channel.GFP], key=sorting_key)
        gfp = np.array([im.data for im in gfp_imgs])
        gfp_norm = _preprocess_gedi_rfp(gfp)
        gfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.2) for frame in gfp_norm])

        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(gfp_filtered)

        for chan, collection in chans.items():

            data_imgs = sorted(collection, key=sorting_key)
            data = np.array([im.data for im in data_imgs])
            data_registered = sr.transform_stack(data, tmats=tmats)
            data_cropped = crop(data_registered)
            data_rescaled = exposure.rescale_intensity(data_cropped, out_range=out_range)

            outpath = format_path(scratch_path, legacy, vertex, mosaic, chan)
            os.makedirs(outpath.parent, exist_ok=True)
            tifffile.imwrite(outpath, data_rescaled)