from collections import defaultdict
from collections.abc import Callable
from typing import Hashable, Iterable, TypeVar


T = TypeVar("T")
G = TypeVar("G", bound=Hashable)
def groupby(it: Iterable[T], keyfunc: Callable[[T], G]) -> dict[G,list[T]]:
    groups = defaultdict(list)
    for e in it:
        groups[keyfunc(e)].append(e)
    return groups