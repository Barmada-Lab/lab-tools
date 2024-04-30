from itertools import product
import pathlib as pl

import click
import xarray as xr

from cytomancer.io.legacy_loader import load_legacy, load_legacy_icc
from cytomancer.io.nd2_loader import load_nd2
from cytomancer.io.lux_loader import load_lux
from cytomancer.io.cq1_loader import load_cq1
from cytomancer.experiment import Axes, ExperimentType


def load_experiment(path: pl.Path, experiment_type: ExperimentType, fillna: bool = False) -> xr.Dataset:
    match experiment_type:
        case ExperimentType.LEGACY:
            return load_legacy(path, fillna)
        case ExperimentType.LEGACY_ICC:
            return load_legacy_icc(path, fillna)
        case ExperimentType.ND2:
            return load_nd2(path)
        case ExperimentType.LUX:
            return load_lux(path, fillna)
        case ExperimentType.CQ1:
            return load_cq1(path)


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
