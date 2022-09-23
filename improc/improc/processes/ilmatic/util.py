from collections.abc import Iterable
from dataclasses import dataclass
import pathlib

@dataclass(eq=True, frozen=True)
class FileSnapshot:
    path: pathlib.Path
    modified_time: int

def snapshot(paths: Iterable[pathlib.Path]) -> set[FileSnapshot]:
    return set([FileSnapshot(path, path.stat().st_mtime_ns) for path in paths])

def get_new(before: set[FileSnapshot], after: set[FileSnapshot]) -> list[pathlib.Path]:
    return sorted(snap.path for snap in after - before)
