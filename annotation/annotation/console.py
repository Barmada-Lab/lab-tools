from pathlib import Path

from .annotation import annotate

def run():
    annotate(Path.cwd())
