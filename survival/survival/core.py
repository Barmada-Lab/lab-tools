from pathlib import Path
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from multiprocessing import Pool

from skimage import exposure, morphology
import numpy as np
from pystackreg import StackReg
import itertools
import pandas as pd
import json
import tifffile
import napari
import os

from improc.experiment.types import Channel, Exposure, Timepoint, Vertex, Dataset, Axis, Image, Mosaic
from improc.processes import Task, Pipeline, BaSiC, Stitch, Stack, TaskError
from improc.experiment import Experiment, loader
from improc.processes.stack import composite_stack, crop
from skimage import filters, measure

from . import gedi
from . import segmentation


def preprocess(experiment: Experiment, gedi: bool):
    elems: list[Task] = [
        BaSiC(),
    ]
    if not gedi:
        elems.append(Stack())
    pipeline = Pipeline(*elems)
    pipeline.run(experiment, "raw_imgs")

@dataclass
class SurvivalResult:
    well: str
    death_pct: float

def calc_avg_obj_size(segmented_stack: np.ndarray):
    sizes = []
    for frame in segmented_stack:
        for props in measure.regionprops(measure.label(frame)):
            sizes.append(props.area)
    if len(sizes) > 0:
        return sum(sizes) / len(sizes)
    else:
        return 0

def event_survival_gedi_gfp(
        stacked: np.ndarray, 
        mask_output: Path | None = None) -> pd.DataFrame | None:

    """
    Utilizes gedi-rfp event signaling alongside gfp-based segmentation to approximate
    survival results in a time-to-event format.

    The survival results are noted as approximate because they utilize an area-based 
    approach to quantify total cell count. The area-based quantification is calculated 
    as follows:

    1. segment somas in GFP
    2. filter out clumped cells from the soma segmentation
    3. calculate the average area of a single cell from the filtered soma segmentation
    4. determine the approximate number of cells at t0 by dividing the area of the 
        (unfiltered) soma segmentation by the average single-cell area.

    From the approximate cell total, the time-to-event quantification is calculated as 
    follows:

    1. segment gedi signal in RFP
    2. for each cell observed in RFP, calculate the average signal intensity over the 
        cell's area
    3. calculate the mean deviation of the average signal intensity
    4. death observations are recorded based a mean deviation greater than an 
        empirically- determined threshold

    After quantifying death, a number of psuedo-observations are recorded to right-censor
    cell survivorship. The number of records is the number of cells at t0 minus the total 
    number of death observations. These psuedo-observations are recorded as being alive at
    the last timepoint.

    The resulting dataframe containing death and psuedo-survival events is then returned,
    ready for cox hazard analysis.
    """

    gfp = stacked[0]
    rfp = stacked[1]

    if 0 in gfp.shape or 0 in rfp.shape:
        return None

    segmented = segmentation.segment_stack(gfp)

    # df = pd.DataFrame(columns=["n_t0", "n_dead"]).rename_axis("id")

    # deaths = gedi.label_deaths(rfp, min_dia=8)

    # df.loc[0] = {
    #     "n_t0": np.count_nonzero(segmented) / 90,
    #     "n_dead": deaths.max()
    # }

    labeled = segmentation.label_segmented_stack(segmented)
    dead = gedi.label_deaths(rfp)

    if mask_output is not None:
        output = np.array((segmented, labeled, dead))
        tifffile.imwrite(mask_output, output)

    avg_obj_size = calc_avg_obj_size(labeled)
    if avg_obj_size == 0:
        print("WARN - avg cell size == 0; check for segmentation error")
        return None

    df = pd.DataFrame(columns=["tp", "dead"]).rename_axis("id")
    n_t0 = int(np.count_nonzero(segmented[0]) / avg_obj_size)
    # if n_t0 < 20:
    #     print("WARN - n_t0 < 20; skipping")
    #     return None

    tf = len(segmented) - 1

    for id in range(n_t0):
        df.loc[id] = {"tp": tf, "dead": 0}

    for tp, frame in enumerate(dead):
        for props in measure.regionprops(frame):
            label = props.label - 1
            if label not in df.index or df.loc[label]["dead"] == 0:
                df.loc[label] = {"tp": tp, "dead": 1}

    return df


def event_survival_gfp(stacked: np.ndarray, stacks_output: Path | None = None) -> pd.DataFrame | None:
    """
    Utilizes a rough area-based cell count quantification to approximate survival 
    results.

    This approach is similar to ::event_survival_gfp in its utilization of average 
    cell area alongside total segmented area to approximate cell counts. This 
    approach does not utilize additional death information; it simply characterizes 
    death as a decrease in observed cells.

    Area is not an ideal estimator for cell count, because cell morphology will often 
    change over the course of an experiment. Cells shrink and swell, stretch, and 
    squeeze together, and frequently do so together across a sample. Therefore, 
    the average cell size must be calculated at each timepoint (as compared to across
    all timepoints,) and consequently suffers from higher variance.

    This higher variance in average cell size can sometimes lead to apparent increases 
    in the total number of observed cells. These increases are ignored by the algorithm. 
    Only decreases between timepoints are recorded. Ideally this hack will be obscured by
    the law of large numbers, but be cognizant of its presence.

    """
    segmented = np.array([segmentation.segment_soma_iN_gfp(frame, 6, .2) for frame in stacked])
    labeled = segmentation.label_segmented_stack(segmented)

    if stacks_output is not None:
        tifffile.imwrite(stacks_output, segmented)

    counts = []
    for total_area, lone_cells in zip(segmented, labeled):
        area = np.count_nonzero(total_area)
        lone_cell_sizes = [props.area for props in measure.regionprops(lone_cells)]
        avg_cell_size = sum(lone_cell_sizes) / len(lone_cell_sizes) if len(lone_cell_sizes) > 0 else 0

        if avg_cell_size > 0:
            counts.append(int(area / avg_cell_size))
        else:
            counts.append(0)

    n_t0 = counts[0]
    tf = len(counts) - 1

    df = pd.DataFrame(columns=["tp", "dead"]).rename_axis("id")
    for id in range(n_t0):
        df.loc[id] = {"tp": tf, "dead": 0}

    total_dead = 0
    for last_tp, (last, now) in enumerate(zip(counts[:-1], counts[1:])):
        delta = last - now
        if delta > 0:
            for id in range(total_dead, total_dead + delta):
                df.loc[id] = {"tp": last_tp + 1, "dead": 1}
            total_dead += delta
    
    return df


def calc_single_cell_survival_gedi(stacked: np.ndarray, stacks_output: Path | None = None) -> pd.DataFrame:
    gfp = stacked[0]
    rfp = stacked[1]

    death_signal = gedi._preprocess_gedi_rfp(rfp)
    death_segmentation = gedi.segment_death_stack(death_signal)

    segmented = np.array([segmentation.segment_soma_iN_gfp(frame, 6, .2) for frame in gfp])
    labeled = segmentation.label_segmented_stack(segmented)

    if stacks_output is not None:
        tifffile.imwrite(stacks_output, labeled)

    df = pd.DataFrame(columns=["tp", "dead"]).rename_axis("id")
    for id in np.unique(labeled):
        df.loc[id] = {"tp": 0, "dead": 0}

    for tp, frame in enumerate(labeled):
        for props in measure.regionprops(frame):
            label = props.label - 1
            if df.loc[label]["dead"] == 1:
                continue

            label_mask = frame == label
            label_death_signal = label_mask * death_segmentation[tp]
            dead = 1 if label_death_signal.any() else 0
            df.loc[label] = {"tp": tp, "dead": dead}
            
    return df

def calc_single_cell_survival_gfp(stacked: np.ndarray, stacks_output: Path | None = None) -> pd.DataFrame:
    segmented = np.array([segmentation.segment_soma_iN_gfp(frame, 6, .2) for frame in stacked])
    labeled = segmentation.label_segmented_stack(segmented)

    if stacks_output is not None:
        tifffile.imwrite(stacks_output, labeled)

    df = pd.DataFrame(columns=["tp", "dead"]).rename_axis("id")
    for id in np.unique(labeled):
        df.loc[id] = {"tp": 0, "dead": 0}

    for tp, frame in enumerate(labeled):
        for id in df.index:
            if df.loc[id]["dead"] == 0 and df not in frame:
                df.loc[id] = {"tp": tp, "dead": 1}

    return df

def make_stacks(experiment: Experiment) -> Iterable[tuple[str, np.ndarray | None]]:
    stitched = experiment.datasets["basic_corrected"]
    
    wells: set[str] = set()
    last_tp = 0
    groups = defaultdict(list)
    for im in stitched.images:
        tp = im.get_tag(Timepoint).index # type:ignore
        mosaic = "_".join(map(str,im.get_tag(Mosaic).index)) # type:ignore
        vertex = f"{im.vertex}_{mosaic}" # type: ignore
        groups[(vertex, tp)].append(im)
        if tp > last_tp:
            last_tp = tp
        wells.add(vertex)

    for well in wells:
        chans = defaultdict(list)
        for tp in range(last_tp + 1):
            ordered = sorted(groups[(well, tp)], key=lambda im: im.get_tag(Exposure).channel)
            for img in ordered:
                chan = img.get_tag(Exposure).channel
                chans[chan].append(img)

        gfp = chans[Channel.GFP]
        rfp = chans[Channel.RFP]

        gfp_raw = np.array([im.data for im in gfp])
        rfp_raw = np.array([im.data for im in rfp])

        gfp_norm = gedi._preprocess_gedi_rfp(gfp_raw)
        gfp_thresh = [frame > np.percentile(frame, 99) for frame in gfp_norm]
        se = morphology.disk(4)
        gfp_opened = np.array([morphology.binary_opening(frame, se) for frame in gfp_thresh])

        registered = composite_stack(np.array((gfp_raw, rfp_raw)), register_on=gfp_opened)

        yield well, registered

def analysis(args):
    stack_loc, mask_output = args
    stacked = tifffile.imread(stack_loc)
    vertex = stack_loc.name.replace(".tif","")
    if stacked is None:
        return vertex, None, None

    df = event_survival_gfp(stacked[0], mask_output / f"{vertex}.tif") # type: ignore
    return vertex, stacked, df

def process(exp_path: Path, scratch_path: Path, save_stacks: bool, save_masks: bool, single_cell: bool, use_gedi: bool, cpus: int):
    experiment = loader.load_experiment(exp_path, scratch_path)

    results_path = exp_path / "results"
    os.makedirs(results_path, exist_ok=True)

    if "basic_corrected" not in experiment.datasets:
        preprocess(experiment, use_gedi)
    
    stacked_output = scratch_path / "stacked"
    if not stacked_output.exists() or len(list(stacked_output.glob("*.tif"))) == 0:
        os.makedirs(stacked_output, exist_ok=True)
        for vertex, img in make_stacks(experiment):
            if img is not None:
                tifffile.imwrite(stacked_output / f"{vertex}.tif", img)
                print(f"wrote {vertex}")
            else:
                print(f"couldn't process {vertex}")

    mask_output = None
    if save_masks:
        mask_output = scratch_path / "masks"
        os.makedirs(mask_output, exist_ok=True)

    dfs = []
    if use_gedi and single_cell:
        # for well, stacked in stack_gfp_gedi(experiment, save_stacks):
        #     df = calc_single_cell_survival_gedi(stacked)
        #     if df is not None:
        #         df["well"] = well
        #         dfs.append(df)
        #         print(f"processed {well}")
        #     else:
        #         print(f"failed to process {well}")
        return
    elif use_gedi and not single_cell:

        with Pool(cpus) as p:
            for vertex, stacked, df in p.imap_unordered(analysis, zip(stacked_output.glob("*.tif"), itertools.repeat(mask_output))):
                if df is not None:
                    df["well"] = vertex
                    dfs.append(df)
                    print(f"processed {vertex}")
                else:
                    print(f"failed to process {vertex}")

    elif not use_gedi and single_cell:
        for im in experiment.datasets["stacked"].images:
            if im.get_tag(Exposure).channel != Channel.GFP: # type: ignore
                continue
            df = calc_single_cell_survival_gfp(im.data, mask_output / f"{im.vertex}.tiff") # type: ignore
            if df is not None:
                df["well"] = im.vertex # type: ignore
                dfs.append(df)
                print(f"processed {im.vertex}")
            else:
                print(f"failed to process {im.vertex}")

    elif not use_gedi and not single_cell:
        for im in experiment.datasets["stacked"].images:
            if im.get_tag(Exposure).channel != Channel.GFP: # type: ignore
                continue
            df = event_survival_gfp(im.data, mask_output / f"{im.vertex}.tiff") # type: ignore
            if df is not None:
                df["well"] = im.vertex # type: ignore
                dfs.append(df)
                print(f"processed {im.vertex}")
            else:
                print(f"failed to process {im.vertex}")

    pd.concat(dfs).to_csv(results_path / "survival.csv")
    

def verify(exp_path: Path, scratch_path: Path):
    experiment = loader.load_experiment(exp_path, scratch_path)

    mask_pairs = defaultdict(list)
    for im in experiment.datasets["masks"].images:
        vertex = im.vertex
        mask_pairs[vertex].append(im)
    
    for im in experiment.datasets["stacked"].images:
        vertex = im.vertex
        mask_pairs[vertex].append(im)

    for vertex, pair in mask_pairs.items():
        if len(pair) != 2:
            continue
        masks, stack = pair
        segmented, labeled, dead = masks.data
        viewer = napari.view_image(stack.data)
        viewer.add_labels(segmented)
        viewer.add_labels(labeled)
        viewer.add_labels(dead)
        napari.run()