import numpy as np
from skimage import measure, morphology
from scipy.stats import norm

def max_entropy(data):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    cdf = data.astype(np.float).cumsum() # type: ignore

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return threshold

def _preprocess_gedi_rfp(stack: np.ndarray):
    def normalize_frame(frame: np.ndarray):
        l, h = np.percentile(frame, (0.5, 99.5))
        mu, std = norm.fit(frame[np.logical_and(frame > l, frame < h)])
        return (frame - mu) / std

    normalized = np.array([normalize_frame(frame) for frame in stack])
    return normalized

def segment_death_stack(stack: np.ndarray, min_dia: int = 6, min_area: int = 20, max_area: int = 180):

    normalized = _preprocess_gedi_rfp(stack)
    thresholded = np.array([frame > np.percentile(frame, 99) for frame in normalized])
    se = morphology.disk(min_dia / 2)
    dead = np.array([morphology.opening(frame, se) for frame in thresholded])
    return dead

def label_deaths(
    gedi_stack: np.ndarray, 
    min_dia: int = 6, 
    min_area: int = 36,
    max_area: int = 180,
    max_frames: int = 4):

    segmented = segment_death_stack(gedi_stack, min_dia, min_area, max_area)
    labeled = measure.label(segmented)
    return labeled