from dataclasses import dataclass
from typing import Generic, TypeVar

import abc


V = TypeVar("V")
E = TypeVar("E")
R = TypeVar("R")

class NoWrappedValue(Exception):
    ...

class Ok:
    ...

class Result(Generic[V,E]):

    @property
    @abc.abstractmethod
    def value(self) -> V:
        ...


@dataclass
class Value(Result[V,E]):
    _value: V

    @property
    def value(self) -> V:
        return self._value


@dataclass
class Error(Result[V,E]):
    _error: E

    @property
    def value(self) -> V:
        raise NoWrappedValue("Error classes contain no value")
