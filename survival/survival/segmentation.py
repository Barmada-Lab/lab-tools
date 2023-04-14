from skimage import filters, measure
from skimage import exposure
from skimage import morphology
import numpy as np
from . import gedi

def segment_soma_iN_gfp(normalized: np.ndarray, min_dia: int = 8, cutoff_freq: float=0.05):
    """
    Segments soma in images typical of iNeurons in GFP

    image - flatfielded, grayscale image
    """
    # filtered = filters.butterworth(image, cutoff_frequency_ratio=cutoff_freq, high_pass=False) # type: ignore

    # l, h = np.percentile(filtered, (0.5,99.5)) 
    # rescaled = exposure.rescale_intensity(filtered, in_range=(l,h))  # type: ignore
    # thresh = filters.threshold_otsu(rescaled) # type: ignore
    # crude_mask = rescaled > thresh

    se = morphology.disk(min_dia / 2)
    h = np.percentile(normalized, 99)
    segmented = morphology.binary_opening(normalized > h, footprint=se) 

    return segmented

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
    normalized = gedi._preprocess_gedi_rfp(stack)
    return np.array([segment_soma_iN_gfp(frame) for frame in normalized])

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