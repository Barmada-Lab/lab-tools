from skimage import filters, measure
from skimage import exposure
from skimage import morphology
import numpy as np
from . import gedi

def logmax_filter(
        frame: np.ndarray, 
        min_sigma: float = 3, 
        max_sigma: float = 7, 
        num_sigma: int = 10):
    """
    Applies a multiscale Laplacian of Gaussian filter stack to an image and returns 
    the maximum response
    """
    assert frame.ndim == 2, f"frame must be 2D; shape is {frame.shape}"
    padding = int(max_sigma)
    padded = np.pad(frame, padding, mode='edge')
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    filter_stack = np.array([filters.laplace(filters.gaussian(padded, sigma=sigma)) for sigma in sigmas])
    unpadded = filter_stack[:, padding:-padding, padding:-padding]
    return unpadded.max(axis=0)

def segment_soma_iN_gfp(
        frame: np.ndarray, 
        min_dia: int = 12, 
        max_dia: int = 30,
        min_area: int = 114,
        max_area: int = 288):
    """
    Segments soma in images typical of iNeurons in GFP

    image - flatfielded, grayscale image
    """

    lap = logmax_filter(frame, min_dia / 2, max_dia / 2)
    thresh = filters.threshold_otsu(lap)
    mask = lap > thresh
    opened = morphology.binary_opening(mask, morphology.disk(min_dia // 2))
    labeled = measure.label(opened)
    for props in measure.regionprops(labeled):
        if props.area < min_area or props.area > max_area:
            opened[labeled == props.label] = 0

    return opened

def label_segmented_stack(stack: np.ndarray, min_area: int = 36, max_area: int = 180):
    labels: np.ndarray = measure.label(stack)
    last_frame = np.unique(labels[0])
    for frame in labels:
        for props in measure.regionprops(frame):
            if props.area < min_area or props.area > max_area:
                frame[frame == props.label] = 0

            # filter discontinuous objects
            if measure.label(frame == props.label).max() > 1: 
                frame[frame == props.label] = 0

            # ignore new and reappearing objects
            if props.label not in last_frame:
                frame[frame == props.label] = 0

        last_frame = np.unique(frame)

    return labels

def segment_stack(stack: np.ndarray):
    return np.array([segment_soma_iN_gfp(frame) for frame in stack])

def filter_lone_cells(img: np.ndarray, min_area: int = 36, max_area: int = 180):
    frame = img.copy()

    shape = frame.shape
    n = int(0.05 * shape[0])
    m = int(0.05 * shape[1])
    censor_mask = np.zeros(shape, dtype=bool)
    censor_mask[:n,:] = True
    censor_mask[-n:,:] = True
    censor_mask[:,:m] = True
    censor_mask[:,-m:] = True

    labeled =  measure.label(frame)
    for props in measure.regionprops(labeled):
        if (censor_mask * (labeled == props.label)).any():
            print("filtering border")
            frame[labeled == props.label] = 0

    return frame