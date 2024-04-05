from itertools import product
import pathlib as pl

import click
import xarray as xr

from .experiment import Axes, ExperimentType


def apply_ufunc_xy(
        func,
        arr: xr.DataArray,
        ufunc_kwargs={},
        **kwargs):
    return xr.apply_ufunc(
        func,
        arr,
        input_core_dims=[[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X]],
        dask="parallelized",
        vectorize=True,
        kwargs=ufunc_kwargs,
        **kwargs)


def iter_idx_prod(arr: xr.DataArray | xr.Dataset, ignore_dims=[]):
    """
    Iterates over the product of an array's indices. Can be used to iterate over
    all the (coordinate-less) XY(Z) planes in an experiment.
    """
    indices = [name for name in arr.indexes if name not in ignore_dims]
    idxs = list([arr.indexes[name] for name in indices])
    for coords in product(*idxs):
        selector = dict(zip(indices, coords))
        yield arr.sel(selector)


def experiment_path_argument(**kwargs):
    return click.argument(
        "experiment-path",
        type=click.Path(exists=True, file_okay=False, path_type=pl.Path),
        **kwargs)


def experiment_type_argument(**kwargs):
    return click.argument(
        "experiment-type",
        type=click.Choice(ExperimentType.__members__),  # type: ignore
        callback=lambda c, p, v: getattr(ExperimentType, v) if v else None, 
        **kwargs)
