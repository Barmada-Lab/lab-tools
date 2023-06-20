from typing import Callable

import numpy as np

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