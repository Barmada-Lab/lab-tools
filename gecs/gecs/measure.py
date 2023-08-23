from typing import Callable, Any
from pathlib import Path
import os

from skimage import exposure # type: ignore
from PIL import Image
import numpy as np
import pandas as pd

def measure(labeled: np.ndarray, raw: np.ndarray, f: Callable[[np.ndarray], Any]) -> dict[int, Any]:
    ids = np.unique(labeled[np.where(labeled != 0)])
    measurements = {}
    for id in ids:
        measurement = f(raw[np.where(np.equal(labeled, id))])
        measurements[id] = measurement
    return measurements

def avg(labeled: np.ndarray, raw: np.ndarray):
    return measure(labeled, raw, np.mean)

def median(labeled: np.ndarray, raw: np.ndarray):
    return measure(labeled, raw, np.median)

def std(labeled: np.ndarray, raw: np.ndarray):
    return measure(labeled, raw, np.std)

def area(labeled: np.ndarray, raw: np.ndarray):
    return measure(labeled, raw, np.count_nonzero)

def cumhist(labeled: np.ndarray, raw: np.ndarray):
    def f(x: np.ndarray):
        rescaled = exposure.rescale_intensity(x, out_range="uint8")
        hist = np.histogram(rescaled, bins=256, range=(0, 255))[0]
        return np.cumsum(hist)
    return measure(labeled, raw, f)

def translate_tups_to_scalars(arr: np.ndarray) -> np.ndarray:
    ids_with_zeros = arr[np.where(~np.equal(arr, np.zeros(arr.shape[-1])).all(axis=-1))]
    ids = np.unique(ids_with_zeros, axis=0)
    labeled = np.zeros(arr.shape[:-1], dtype=np.uint16)
    for idx, id in enumerate(ids):
        labeled[np.equal(arr, id).all(axis=-1)] = idx + 1
    return labeled

def measure_rois(
        raw_path: Path, 
        segmented_path: Path, 
        measure_avg: bool = False, 
        measure_median: bool = False, 
        measure_std: bool = False, 
        measure_area: bool = False,
        measure_cumhist: bool = False):

    raw_pil = Image.open(raw_path)
    raw = np.array(raw_pil)

    segmented_pil = Image.open(segmented_path)
    segmented = np.array(segmented_pil)

    if segmented.ndim == 3 and segmented.shape[-1] == 3:
        segmented = translate_tups_to_scalars(segmented)

    ids = np.unique(segmented[np.where(segmented != 0)])

    measurements = pd.DataFrame(index=ids)
    if measure_avg:
        avg_measurements = avg(segmented, raw)
        measurements['avg'] = pd.Series(avg_measurements)
    if measure_median:
        median_measurements = median(segmented, raw)
        measurements['median'] = pd.Series(median_measurements)
    if measure_std:
        std_measurements = std(segmented, raw)
        measurements['std'] = pd.Series(std_measurements)
    if measure_area:
        area_measurements = area(segmented, raw)
        measurements['area'] = pd.Series(area_measurements)
    if cumhist:
        cumhist_measurements = cumhist(segmented, raw)
        measurements['cumhist'] = pd.Series(cumhist_measurements)

    return measurements

def cli_entry(args):
    roi_paths = list(args.roi_dir.glob('*'))
    if roi_paths[0].name.endswith('.png'):
        raw_paths = [args.raw_dir / roi.name.replace(".png",".tif") for roi in roi_paths]
    else:
        raw_paths = [args.raw_dir / roi.name for roi in roi_paths]

    assert all([roi_path.exists() for roi_path in roi_paths]), 'Not all raw images have a corresponding ROI image'

    df = pd.DataFrame()
    for raw, roi in zip(raw_paths, roi_paths):
        measurements = measure_rois(raw, roi, args.avg, args.median, args.std, args.area, args.cumhist)
        measurements.insert(loc=0, column='img', value=raw.name)
        df = pd.concat([df, measurements], axis=0)

    output_path = args.output if args.output is not None else Path.cwd() / 'measurements.csv'
    df.to_csv(output_path, index_label='roi_id')