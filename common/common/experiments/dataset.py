from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Type

from common.experiments.image import Image
from common.experiments.tags import Tag

@dataclass
class Dataset:
    label: str
    images: list[Image]

def select_by_tags(images: list[Image], *ts: Type[Tag]) -> Iterator[Image]:
    for img in images:
        if all(map(lambda tag: img.get_tag(tag) is not None, ts)):
            # If all the tags are present in the image, we yield it.
            yield img

def filter_tags(images: list[Image], *ts: Type[Tag]) -> Iterator[Image]:
    for img in images:
        if all(map(lambda tag: img.get_tag(tag) is None, ts)):
            # If all the tags are present in the image, we yield it.
            yield img

def aggregate(images: list[Image], t: Type[Tag]) -> list[list[Image]]:
    """ Aggregate based on similarity of a given tag """
    groups = defaultdict(list[Image])
    for img in images:
        group_tag = img.get_tag(t)
        groups[group_tag].append(img)

    return list(groups.values())

def contra_aggregate(images: list[Image], t: Type[Tag]) -> list[list[Image]]:
    """ Aggregate based on dissimilarity of a given tag """
    groups = defaultdict(list[Image])
    for img in images:
        group_tag = img.get_tag(t)
        if group_tag is None:
            continue
        sans = tuple(tag for tag in img.meta.tags if tag != group_tag)
        groups[sans].append(img)

    return list(groups.values())
