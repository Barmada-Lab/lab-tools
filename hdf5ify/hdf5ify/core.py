from collections import namedtuple

from PIL import Image
from PIL.TiffTags import TAGS

import numpy as np
import h5py
import glob
import os

from common import utils

Args = namedtuple("Args", ["src", "dest"])

def print_usage(program_name: str):
    print(f"Usage: {program_name} SRC DEST")

def parse_args(args: list[str]) -> Args:
    if not len(args) >= 3:
        print_usage(args[0])
        raise Exception("not enough arguments")
    src = os.path.relpath(args[1])
    dest = os.path.relpath(args[2])
    if not os.path.isdir(src):
        raise Exception(f"invalid SRC directory: {src}")
    return Args(src, dest)

def hdf5ify(src, dest):

    tifs = os.path.join(src,"raw_imgs/**/*.tif")
    with h5py.File(dest, "w") as output:

        config = utils.find_config_csv(src)
        if config is None:
            print(f"no config found in {src}")
            return

        metas = [utils.extract_meta(path) for path in glob.glob(tifs, recursive=True)]
        if len(metas) == 0:
            print(f"no files found in {src}")
            return

        with Image.open(metas[0].path) as img:
            tags = {TAGS[key]: value for key, value in img.getexif().items()}
            (img_height, img_width) = np.shape(img)

        montage_dim = int(config['Montage XY'])
        layout_indexing = utils.determine_layout_indexing(config["microscope"], montage_dim)

        for meta in metas:
            group = output.require_group(f"{meta.col}_{meta.row}/{meta.channel}")
            dtype = np.array(img).dtype
            dataset = group.require_dataset(
                "montage",
                (montage_dim, montage_dim, img_height, img_width),
                dtype=dtype,
                compression="gzip",
                compression_opts=1
            )
            (x, y) = np.where(layout_indexing == meta.montage_idx)
            (x, y) = (x[0], y[0]) # extract the coordinates to index the h5py dataset

            with Image.open(meta.path) as img:
                dataset[x,y,:,:] = np.array(img)
