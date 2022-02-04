import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from PIL import Image

def generate_background(paths: list[pathlib.Path], output: pathlib.Path,
        display=False, force=False):

    images = np.array([np.array(Image.open(path)) for path in paths])
    median = np.median(images, axis=0, overwrite_input=True)

    if display:
        plt.imshow(median)
        plt.show()
        return

    if not force and os.path.exists(output):
        while selection := input(f"{output} exists; overwrite? confirm [y/N]"):
            match selection.lower():
                case "\n" | "n":
                    return
                case "y":
                    break
                case _:
                    print("I'm afraid I can't do that, Dave.")

    Image.fromarray(median.astype(np.uint16)).save(output)

def subtract(background_path, input_path, output_base, ftype, display=False, force=False):

    if not os.path.isfile(background_path):
        print(f"input path {background_path} does not exist.")
        print("exiting...")
        return

    if not os.path.exists(input_path):
        print(f"input path {input_path} does not exist.")
        print("exiting...")
        return

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    paths = []
    if os.path.isfile(input_path):
        paths.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for _file in filter(lambda f: f.endswith(ftype), files):
                paths.append(os.path.join(root, _file))

    # Reinterpret images as int32 to avoid underflow when subtracting
    background = np.array(Image.open(background_path)).astype(np.int32)

    for path in paths:
        print(path)
        img = np.array(Image.open(path)).astype(np.int32)
        subtracted = img - background
        subtracted[subtracted < 0] = 0
        subtracted = subtracted.astype(np.uint16) # Convert back to 16 bit after truncation

        mirror_path = os.path.relpath(path, input_path) if path != input_path else os.path.basename(path)
        output_path = os.path.join(output_base, mirror_path)
        mirror_parent = os.path.abspath(os.path.join(output_path, os.path.pardir))
        print(mirror_parent)
        if not os.path.isdir(mirror_parent):
            os.makedirs(mirror_parent)

        if display:
            plt.imshow(subtracted)
            plt.show()
            continue

        if (not force) and os.path.exists(output_path):
            print(f"{output_path} exists; either delete the output or re-run subtract with -f to overwrite")
            return

        Image.fromarray(subtracted).save(output_path)
