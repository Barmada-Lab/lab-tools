import abc
from dataclasses import dataclass
import pathlib

from improc.common.result import Result, Error, Value

from .types import Experiment
from .legacy.loader import LegacyLoader


@dataclass
class UnknownLayout:
    path: pathlib.Path

LoadError = UnknownLayout

def load_experiment(path: pathlib.Path, scratch_dir: pathlib.Path) -> Result[Experiment, LoadError]:
    """
    auto-detects experiment structure and loads accordingly
    """
    loader = None
    if loader is None:
        loader = LegacyLoader(path, scratch_dir)
        return Value(loader.load())

    #default
    return Error(UnknownLayout(path))
