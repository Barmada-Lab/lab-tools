import pathlib as pl

import xarray as xr

from .legacy_loader import load_legacy, load_legacy_icc
from .nd2_loader import load_nd2
from .lux_loader import load_lux
from ..experiment import ExperimentType


def load_experiment(path: pl.Path, experiment_type: ExperimentType, fillna: bool = True) -> xr.Dataset:
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
            raise NotImplementedError("CQ1 loader not implemented yet")
