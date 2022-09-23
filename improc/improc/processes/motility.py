from pystackreg import StackReg
from numpy.lib.stride_tricks import sliding_window_view
from skimage.exposure import rescale_intensity
from skimage import filters
from skimage import measure
from skimage.morphology import disk, dilation, white_tophat

import numpy as np

def diff(arr):
    a = filters.gaussian(arr[0,0,-1], sigma=1.5)
    b = filters.gaussian(arr[0,0,0], sigma=1.5)
    footprint = disk(2)
    diff = a - dilation(b, footprint)
    diff[diff < 0] = 0
    thresh = filters.threshold_otsu(diff)
    diff[diff < thresh] = 0
    return diff

def hilight_motion(stack, window=10):
    """ Make sure you register before calling this function """
    sliding = sliding_window_view(stack, window_shape=(window, stack.shape[1], stack.shape[2]))
    subd = np.array(list(map(diff, sliding)))
    return subd

def calc_motility(stack):
    # filter noise
    l, h = np.percentile(stack, (0.5, 99.5))
    rescaled = rescale_intensity(stack, in_range=(l,h))

    # handle still frame
    footprint = disk(3)
    smooth = filters.gaussian(rescaled[0], sigma=1.5)
    tophatted = white_tophat(smooth, footprint)
    still_thresh = filters.threshold_otsu(tophatted)
    threshd = np.zeros_like(tophatted)
    threshd[tophatted > still_thresh] = 1

    hilighted = hilight_motion(rescaled)

    labels = measure.label(hilighted.astype(np.bool8))
    props = measure.regionprops(labels)
    avg_mito_area = np.median([prop.area for prop in props])

    hilighted_area_per_frame = np.count_nonzero(hilighted) / len(hilighted)
    pct = hilighted_area_per_frame / np.count_nonzero(threshd)

    return {
        "pct": pct,
        "counts_per_frame": hilighted_area_per_frame / avg_mito_area,
        "still": threshd,
        "motion_hilighted": hilighted
    }
