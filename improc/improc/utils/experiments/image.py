from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import numpy as np

from .tags import Tag


@dataclass
class ImgMeta:
    tags: list[Tag]

class Image(ABC):

    def __init__(self, meta: ImgMeta) -> None:
        super().__init__()
        self.meta = meta

    @abstractmethod
    def load(self) -> np.ndarray:
        """ Load the content of this image into memory """
        pass

    @abstractmethod
    def save(self, a: np.ndarray, overwrite=False) -> None:
        """ Persist this image """
        pass

    def get_tag(self, t: Type[Tag]) -> Any:
        for tag in self.meta.tags:
            if isinstance(tag, t):
                return tag
        return None

    def replace_tag(self, old: Tag, new: Tag) -> 'Image':
        return self
