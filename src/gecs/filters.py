import numpy as np
import xarray as xr
from skimage import filters # type: ignore

from .experiment import Axes

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
