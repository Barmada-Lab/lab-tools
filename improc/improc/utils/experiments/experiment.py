from abc import ABC, abstractmethod
from dataclasses import dataclass

from .image import Image
from ..types import MFSpec
from .dataset import Dataset


@dataclass
class Experiment:
    label: str
    raw: list[Image]
    mfile: MFSpec | None

class ExperimentLoader(ABC):
    @abstractmethod
    def load(self) -> Experiment:
        pass
