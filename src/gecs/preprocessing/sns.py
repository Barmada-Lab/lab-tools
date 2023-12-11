from pathlib import Path
from collections import defaultdict

from skimage import exposure, filters # type: ignore
from pystackreg import StackReg
import tifffile
import numpy as np
import os
import click

from improc.experiment.types import Vertex, Mosaic, Exposure, Channel, Timepoint, Image, Experiment
from improc.experiment import loader
from improc.processes.types import Pipeline, Task
from improc.processes.flatfield import BaSiC
from improc.processes.stitching import Stitch
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

def stitch_n_stack(experiment: Experiment, collection: str, legacy: bool = False, out_range: str = "uint16", stitch: bool = True):

    steps: list[Task] = [ BaSiC() ]
    if stitch:
        steps.append(Stitch())
    initial_pipeline = Pipeline(*steps)
    initial_pipeline.run(experiment, collection)

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
        chans: dict[Channel, list[Image]] = defaultdict(list)
        for image in images:
            chan = image.get_tag(Exposure).channel
            chans[chan].append(image)
        
        gfp_imgs = sorted(chans[Channel.GFP], key=sorting_key)
        gfp = np.array([im.data for im in gfp_imgs])
        gfp_filtered = np.array([filters.butterworth(frame, high_pass=False, cutoff_frequency_ratio=0.2) for frame in gfp])

        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(gfp_filtered)

        for chan, collection in chans.items(): # type: ignore

            data_imgs = sorted(collection, key=sorting_key)
            data = np.array([im.data for im in data_imgs]) # type: ignore
            data_registered = sr.transform_stack(data, tmats=tmats)
            data_cropped = crop(data_registered)
            data_rescaled = exposure.rescale_intensity(data_cropped, out_range=out_range)

            outpath = format_path(experiment.scratch_dir, legacy, vertex, mosaic, chan)
            os.makedirs(outpath.parent, exist_ok=True)
            tifffile.imwrite(outpath, data_rescaled)

@click.command("stitch-n-stack")
@click.argument('experiment_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--scratch-dir', type=click.Path(path_type=Path), default=None)
@click.option('--collection', type=str, default="raw_imgs")
@click.option('--out-range', type=str, default='uint16')
@click.option('--no-stitch', is_flag=True, default=False)
@click.option('--legacy', is_flag=True, default=False)
def cli_entry(experiment_dir: Path, scratch_dir: Path, collection: str, legacy: bool, out_range: str, no_stitch: bool):
    scratch_dir = scratch_dir if scratch_dir is not None else experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(experiment_dir, scratch_dir)
    stitch_n_stack(
        experiment, 
        collection,
        legacy, 
        out_range, 
        not no_stitch)