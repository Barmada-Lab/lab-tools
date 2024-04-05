import warnings
import logging

import xarray as xr
from skimage import transform as skt
import numpy as np
from pystackreg import StackReg

from .experiment import Axes


logger = logging.getLogger(__name__)

def register(arr: xr.DataArray):
    def _register(stack):
        sr = StackReg(StackReg.RIGID_BODY)
        with warnings.catch_warnings(record=True) as w:
            tmats = sr.register_stack(stack)
            return tmats if len(w) == 0 else np.array([np.eye(3) for _ in range(stack.shape[0])])
    return xr.apply_ufunc(
        _register,
        arr,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, "tmat_y", "tmat_x"]],
        dask_gufunc_kwargs=dict(
            output_sizes={"tmat_y": 3, "tmat_x": 3},
            allow_rechunk=True,
        ),
        dask="parallelized",
        vectorize=True)


def min_bb(arr: xr.DataArray, tmats: xr.DataArray):
    shape = (arr.y.size, arr.x.size)

    def _min_bb(tmats):
        sr = StackReg(StackReg.RIGID_BODY)
        t = tmats.shape[0]
        ones = np.ones((t, *shape))
        transformed = sr.transform_stack(ones, tmats=tmats)
        mask = transformed > 0.5
        proj = np.logical_and.reduce(mask)
        return ones * proj

    return xr.apply_ufunc(
        _min_bb,
        tmats,
        input_core_dims=[[Axes.TIME, "tmat_y", "tmat_x"]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask_gufunc_kwargs=dict(
            output_sizes={Axes.Y: shape[0], Axes.X: shape[1]},
            allow_rechunk=True,
        ),
        output_dtypes=[bool],
        dask="parallelized",
        vectorize=True)


def mask_bb(arr: xr.DataArray, mask: xr.DataArray):
    def _mask_bb(arr, mask):
        response = arr.copy()
        response[~mask] = 0
        return response

    return xr.apply_ufunc(
        _mask_bb,
        arr,
        mask,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X], [Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized",
        vectorize=True)


def transform(arr: xr.DataArray, tmats: xr.DataArray, categorical=False):
    """
    Transform a stack of images using a set of transformation matrices.
    
    Parameters
    ----------
    arr : xr.DataArray
        A data array.
    tmats : xr.DataArray
        An array of transformation matrices.
    categorical : bool
        Set to true for categorical data like segmentation masks.
        If True, the transform will use nearest neighbor interpolation to preserve the categorical nature of the data.
        Otherwise, the transform will use bilinear interpolation.
    """

    def _transform(stack, tmats):
        order = 0 if categorical else 1
        warped = np.array([skt.warp(frame, tmat, order=order, mode="edge") for frame, tmat in zip(stack.copy(), tmats)])
        return warped

    return xr.apply_ufunc(
        _transform,
        arr,
        tmats,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X], [Axes.TIME, "tmat_y", "tmat_x"]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        output_dtypes=[arr.dtype],
        dask="parallelized",
        vectorize=True)
