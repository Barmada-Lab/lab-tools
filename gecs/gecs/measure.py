from typing import Callable
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd

def measure(labeled: np.ndarray, raw: np.ndarray, f: Callable[[np.ndarray], float]) -> dict[int, float]:
    ids = np.unique(labeled[np.where(labeled != 0)])
    measurements = {}
    for id in ids:
        measurement = f(raw[np.where(labeled == id)])
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


def measure_rois(
        raw_path: Path, 
        segmented_path: Path, 
        measure_avg: bool = False, 
        measure_median: bool = False, 
        measure_std: bool = False, 
        measure_area: bool = False):

    raw_pil = Image.open(raw_path)
    raw = np.array(raw_pil)
    
    segmented_pil = Image.open(segmented_path)
    segmented = np.array(segmented_pil)

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

    return measurements

def cli_entry(args):
    pass