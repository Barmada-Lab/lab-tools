import argparse
import pathlib
import sys

from . import core

GENERATE_CMD = "generate"
SUBTRACT_CMD = "subtract"

def main():
    root_parser = argparse.ArgumentParser("flatfield")
    root_parser.add_argument("--display", action="store_true", help="Display the generated image and exit")

    subparsers = root_parser.add_subparsers(required=True, dest="command")

    generate_parser = subparsers.add_parser(GENERATE_CMD, help="Generate a background image")
    generate_parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Input images")
    generate_parser.add_argument("output", type=pathlib.Path, help="Output path for generated image")
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
        core.generate_background(
            paths=args.inputs,
            output=args.output,
            display=args.display)
    elif cmd == SUBTRACT_CMD:
        core.subtract(
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
