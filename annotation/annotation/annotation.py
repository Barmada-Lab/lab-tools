from pathlib import Path

from pybasic import shading_correction
from skimage import filters
import numpy as np
import tifffile
import napari

CHANNELS = [
    "white",
    "DAPI",
    "RFP",
    "GFP"
]

CHANNEL_COLORS = [
    "gray",
    "cyan",
    "red",
    "green"
]

def get_label_path(path: Path) -> Path:
    name = path.name.strip(".ome.tif")
    new_name = name + "_labels.ome.tif"
    return path.parent / new_name


def label_tiff(tiff: Path, label_path: Path) -> None:
    img = np.array(tifffile.imread(tiff))
    viewer = napari.view_image(
        img,
        channel_axis=0,
        name=CHANNELS,
        colormap=CHANNEL_COLORS
    )

    idx: int = CHANNELS.index("GFP")
    gfp_img = img[idx]
    basic = shading_correction.BaSiC(gfp_img)
    basic.prepare()
    basic.run()
    corrected = np.array([basic.normalize(img_slice) for img_slice in gfp_img])

    frame = corrected[27] # should be
    blurred = filters.gaussian(frame, sigma=1)
    thresh = filters.threshold_yen(blurred)

    viewer.add_labels(blurred > thresh)

    napari.run()
    viewer.layers.save(str(label_path), selected=True)

def annotate(directory: Path) -> None:
    for tiff in directory.glob("*.tif"):
        if (label_path := get_label_path(tiff)).exists():
            print(f"{label_path} already exists; skipping")
            continue
        label_tiff(tiff, label_path)
