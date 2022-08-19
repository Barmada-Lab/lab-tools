from abc import ABC, abstractmethod
from dataclasses import dataclass

from common.types import MFSpec

from .dataset import Dataset


@dataclass
class Experiment:
    label: str
    datasets: list[Dataset]
    mfile: MFSpec | None

class ExperimentLoader(ABC):
    @abstractmethod
    def load(self) -> Experiment:
        pass
