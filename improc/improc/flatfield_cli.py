import argparse
import pathlib
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from . import flatfield

GENERATE_CMD = "generate"
SUBTRACT_CMD = "subtract"

def rlist_files(path, ftype):
    for root, _, files in os.walk(path):
        for _file in filter(lambda f: f.endswith(ftype), files):
            yield os.path.join(root, _file)

def generate_background(inputs: list[pathlib.Path], output: pathlib.Path, display=False, force=False, recursive=False):
    paths = []
    for input_path in inputs:
        if os.path.isdir(input_path) and recursive:
            paths += list(rlist_files(input_path, "tif"))
        elif os.path.isdir(input_path) and not recursive:
            print(f"{input_path} is a directory and --recursive is not specified; skipping.")
        elif os.path.isfile(input_path):
            paths.append(input_path)
        else:
            print(f"{input_path} does not exist")
            return

    if not display and not force and os.path.exists(output):
        while selection := input(f"{output} exists; overwrite? confirm [y/N]"):
            if selection.lower() in ["\n", "n"]:
                return
            elif selection.lower() == "y":
                break
            else:
                print("I'm afraid I can't do that, Dave.")

    median = flatfield.background_from_paths(paths)

    if display:
        plt.imshow(median)
        plt.show()
        return

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
        for path in rlist_files(input_path,ftype):
            paths.append(path)

    background = np.array(Image.open(background_path))

    for path in paths:
        print(f"subtracting {path}")
        img = np.array(Image.open(path)).astype(background.dtype)
        subtracted = flatfield.trunc_sub(img, background)

        mirror_path = os.path.relpath(path, input_path) if path != input_path else os.path.basename(path)
        output_path = os.path.join(output_base, mirror_path)
        mirror_parent = os.path.abspath(os.path.join(output_path, os.path.pardir))
        print(f"writing to {mirror_parent}")
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

def main():
    root_parser = argparse.ArgumentParser("flatfield")
    root_parser.add_argument("--display", action="store_true", help="Display the generated image and exit")

    subparsers = root_parser.add_subparsers(required=True, dest="command")

    generate_parser = subparsers.add_parser(GENERATE_CMD, help="Generate a background image")
    generate_parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Input images")
    generate_parser.add_argument("output", type=pathlib.Path, help="Output path for generated image")
    generate_parser.add_argument("-r", "--recursive", action="store_true", help="Use all files in input directory")
    generate_parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    generate_parser.add_argument("--display", action="store_true", help="Display the generated image and exit")

    subtract_parser = subparsers.add_parser(SUBTRACT_CMD, help="Subtract background from images")
    subtract_parser.add_argument("background", type=pathlib.Path, help="Background image to subtract")
    subtract_parser.add_argument("input_path", type=pathlib.Path, help="Input path")
    subtract_parser.add_argument("output_base", type=pathlib.Path, help="Output path")
    subtract_parser.add_argument("--ftype", type=str, choices=["tif"], default="tif", help="filetype to filter for when recursing")
    subtract_parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    subtract_parser.add_argument("--display", action="store_true", help="Display the generated image and exit")

    args = root_parser.parse_args()
    cmd = args.command
    if cmd == GENERATE_CMD:
        generate_background(
            inputs=args.inputs,
            output=args.output,
            recursive=args.recursive,
            display=args.display)
    elif cmd == SUBTRACT_CMD:
        subtract(
            args.background,
            args.input_path,
            args.output_base,
            ftype=args.ftype,
            display=args.display,
            force=args.force,)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
