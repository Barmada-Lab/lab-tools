import numpy as np

from skimage import exposure, filters, feature, segmentation # type: ignore

def multiscale_LoG_filter(img: np.ndarray, start_sigma: float = 1, end_sigma: float = 10, n_sigma: int = 10) -> np.ndarray:
    padded = np.pad(img, int(end_sigma), mode='edge')
    sigmas = np.linspace(start_sigma, end_sigma, n_sigma)
    filtered= np.array([filters.laplace(filters.gaussian(padded, sigma)) for sigma in sigmas])
    return filtered[:, end_sigma:-end_sigma, end_sigma:-end_sigma]

def watershed(frame: np.ndarray, mask: np.ndarray):
    blobs = feature.blob_log(frame, min_sigma=3, max_sigma=15, num_sigma=20, threshold=0.1)
    seed = np.zeros_like(frame)
    for idx, (x, y) in enumerate(blobs[:, :2].astype(int)):
        seed[x, y] = idx + 1

    equalized = exposure.equalize_adapthist(frame, clip_limit=0.1)
    return segmentation.watershed(-equalized, seed, mask=mask)

def segment_puncta_3d(stack: np.ndarray):

    filtered = np.array([multiscale_LoGmax_filter(frame, 3, 15, 20) for frame in stack])
    thresh = filters.threshold_otsu(filtered)
    binary = filtered > thresh

    ws = np.array([watershed(frame, mask) for frame, mask in zip(stack, binary)])
    return ws