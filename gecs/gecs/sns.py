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
from improc.processes import BaSiC, Stitch, Pipeline


def stitch_n_stack(experiment_path: Path, scratch_path: Path, legacy: bool):

    experiment = loader.load_experiment(experiment_path, scratch_path)
    # initial_pipeline = Pipeline(
    #     BaSiC(),
    #     Stitch(),
    # )
    # initial_pipeline.run(experiment, "raw_imgs")

    # now stack based on GFP registration
    locs = defaultdict(list)
    for image in experiment.datasets["stitched"].images:
        vertex = image.get_tag(Vertex)
        locs[vertex].append(image)

    def sorting_key(image):
        return image.get_tag(Timepoint).index

    for vertex, images in locs.items():
        chans = defaultdict(list)
        for image in images:
            chan = image.get_tag(Exposure).channel
            chans[chan].append(image)
        
        gfp_imgs = sorted(chans[Channel.GFP], key=sorting_key)
        gfp = np.array([im.data for im in gfp_imgs])
        gfp_norm = _preprocess_gedi_rfp(gfp)
        gfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.1) for frame in gfp_norm])

        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(gfp_filtered)
        gfp_registered = sr.transform_stack(gfp_filtered, tmats=tmats)
        gfp_rescaled = exposure.rescale_intensity(gfp_registered, out_range=np.uint16)

        rfp_imgs = sorted(chans[Channel.RFP], key=sorting_key)
        rfp = np.array([im.data for im in rfp_imgs])
        rfp_norm = _preprocess_gedi_rfp(np.array(rfp))
        rfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.1) for frame in rfp_norm])
        rfp_registered = sr.transform_stack(rfp_filtered, tmats=tmats)
        rfp_rescaled = exposure.rescale_intensity(rfp_registered, out_range=np.uint16)

        if legacy:
            outpath_gfp = scratch_path / "stacked" / "GFP" / f"{vertex.label}.tif"
            outpath_rfp = scratch_path / "stacked" / "RFP" / f"{vertex.label}.tif"
        else:
            outpath_gfp = scratch_path / "stacked" / f"{vertex.label}.tif"
            outpath_rfp = scratch_path / "stacked" / f"{vertex.label}.tif"

        os.makedirs(outpath_gfp.parent, exist_ok=True)
        os.makedirs(outpath_rfp.parent, exist_ok=True)

        tifffile.imwrite(outpath_gfp, gfp_rescaled)
        tifffile.imwrite(outpath_rfp, rfp_rescaled)