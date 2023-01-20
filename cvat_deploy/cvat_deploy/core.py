import os
import sys
from pathlib import Path

import cv2
import tifffile
import numpy as np
from skimage.exposure import rescale_intensity

def write_to_mp4(arr: np.ndarray, destination: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dims = arr.shape[1:-1][::-1] # get the x/y dims and reverse
    writer = cv2.VideoWriter(str(destination), fourcc, 1, dims)
    for frame in arr:
        writer.write(frame)
    writer.release()

def convert_to_rgb(img: np.ndarray):
    rescaled = rescale_intensity(img, out_range=np.uint8) #type:ignore
    return np.stack((rescaled,)*3, axis=-1)

def tiff_to_mp4(origin: Path, destination: Path):
    tiff = tifffile.imread(origin)
    rgb = np.array(list(map(convert_to_rgb, tiff)))
    write_to_mp4(rgb, destination)

def convert_dir(origin: Path, destination: Path):
    for tiff_file in origin.glob("*.tif"):
        mp4_file = destination / tiff_file.name.replace(".tif", ".mp4")
        tiff_to_mp4(tiff_file, mp4_file)

def main():
    origin = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    os.makedirs(destination, exist_ok=True)
    convert_dir(origin, destination)

if __name__ == "__main__":
    main()
