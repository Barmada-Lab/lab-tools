import warnings
import xarray as xr
import numpy as np
from pystackreg import StackReg

from .experiment import Axes


def register(arr: xr.DataArray):
    def _register(stack):
        sr = StackReg(StackReg.RIGID_BODY)
        tmats = sr.register_stack(stack)
        return tmats
    return xr.apply_ufunc(
        _register,
        arr,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, "tmat_y", "tmat_x"]],
        dask_gufunc_kwargs=dict(
            output_sizes={"tmat_y": 3, "tmat_x": 3},
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
        dask="parallelized",
        vectorize=True)


def _transform_float(arr: xr.DataArray, tmats: xr.DataArray):
    def _transform(stack, tmats):
        sr = StackReg(StackReg.RIGID_BODY)
        transformed = sr.transform_stack(stack, tmats=tmats)
        return transformed
    return xr.apply_ufunc(
        _transform,
        arr,
        tmats,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X], [Axes.TIME, "tmat_y", "tmat_x"]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask="parallelized",
        output_dtypes=[np.float64],
        vectorize=True)


def _transform_bool(arr: xr.DataArray, tmats: xr.DataArray):
    def _transform(stack, tmats):
        sr = StackReg(StackReg.RIGID_BODY)
        transformed = sr.transform_stack(stack, tmats=tmats)
        return transformed > 0.5
    return xr.apply_ufunc(
        _transform,
        arr,
        tmats,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X], [Axes.TIME, "tmat_y", "tmat_x"]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask="parallelized",
        output_dtypes=[bool],
        vectorize=True)


def _transform_int(arr: xr.DataArray, tmats: xr.DataArray):
    def _transform(stack, tmats):
        sr = StackReg(StackReg.RIGID_BODY)
        transformed = sr.transform_stack(stack, tmats=tmats)
        return np.rint(transformed)
    return xr.apply_ufunc(
        _transform,
        arr,
        tmats,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X], [Axes.TIME, "tmat_y", "tmat_x"]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask="parallelized",
        output_dtypes=[int],
        vectorize=True)


def transform(arr: xr.DataArray, tmats: xr.DataArray):
    """ Multiple dispatch? Never heard of it. """
    if np.issubdtype(arr.dtype, np.bool_):
        return _transform_bool(arr, tmats)
    elif np.issubdtype(arr.dtype, np.floating):
        return _transform_float(arr, tmats)
    elif np.issubdtype(arr.dtype, np.integer):
        return _transform_int(arr, tmats)
    else:
        warnings.warn(f"Unhandled transform array of type {arr.dtype}; defaulting to float64")
        return _transform_float(arr, tmats)
