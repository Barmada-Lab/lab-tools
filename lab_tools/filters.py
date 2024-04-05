from skimage import filters, morphology  # type: ignore
import numpy as np
import xarray as xr

from .experiment import Axes
from .util import apply_ufunc_xy


def median(arr: xr.DataArray, footprint: np.ndarray | int, mode='nearest'):

    if isinstance(footprint, int):
        footprint = morphology.disk(footprint)

    return apply_ufunc_xy(
        filters.median,
        arr,
        ufunc_kwargs={"footprint": footprint, "mode": mode},)


def logmax_filter2d(
        arr: xr.DataArray,
        min_sigma: float = 14,
        max_sigma: float = 25,
        n_sigma: int = 3):
    def _logmax_filter2d(frame):
        padding = int(max_sigma)
        padded = np.pad(frame, padding, mode='edge')
        sigmas = np.linspace(min_sigma, max_sigma, n_sigma)
        filter_stack = np.array(
            [filters.laplace(filters.gaussian(padded, sigma=sigma)) for sigma in sigmas])
        unpadded = filter_stack[:, padding:-padding, padding:-padding]
        filtered = unpadded.max(axis=0)
        return filtered
    return xr.apply_ufunc(
        _logmax_filter2d,
        arr,
        input_core_dims=[[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X]],
        dask="parallelized",
        vectorize=True)


def logmax_filter3d(
        arr: xr.DataArray,
        min_sigma: float = 14,
        max_sigma: float = 25,
        n_sigma: int = 3):
    def _logmax_filter3d(frame):
        padding = int(max_sigma)
        padded = np.pad(frame, padding, mode='edge')
        sigmas = np.linspace(min_sigma, max_sigma, n_sigma)
        filter_stack = np.array(
            [filters.laplace(filters.gaussian(padded, sigma=sigma)) for sigma in sigmas])
        unpadded = filter_stack[:, padding:-padding, padding:-padding]
        filtered = unpadded.max(axis=0)
        return filtered
    return xr.apply_ufunc(
        _logmax_filter3d,
        arr,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask="parallelized",
        vectorize=True)
