from skimage import filters, measure # type: ignore
from skimage import exposure # type: ignore
from skimage import morphology
import numpy as np
from . import gedi

def logmax_filter(
        frame: np.ndarray, 
        min_sigma: float = 6, 
        max_sigma: float = 14, 
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
        arr: np.ndarray, 
        min_dia: int = 12, 
        max_dia: int = 30):
    """
    Segments soma in images typical of iNeurons in GFP

    """

    rescaled = exposure.rescale_intensity(arr, out_range="uint16")
    eqd = exposure.equalize_adapthist(rescaled, clip_limit=0.01)
    lap = np.array([logmax_filter(frame, min_dia / 2, max_dia / 2) for frame in eqd])
    thresh = filters.threshold_otsu(lap)
    mask = lap > thresh
    opened = np.array([morphology.binary_opening(frame, morphology.disk(min_dia // 2)).astype(np.uint8) for frame in mask])

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