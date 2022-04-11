import argparse
import pathlib
import sys
import os

from annotation.utils import confirm_overwrite

from . import core
from .constants import IJ_ENC, JSON_ENC, PKL_ENC, AUTO_ENC

def main():
    root = argparse.ArgumentParser("anno", description="annotation utility")
    subparsers = root.add_subparsers(required=True, dest="command")

    convert = subparsers.add_parser("convert", help="convert annotations from one format to another")
    convert.add_argument("src", nargs="+", type=pathlib.Path, help="annotation to convert")
    convert.add_argument("dest", type=pathlib.Path, help="output destination")
    convert.add_argument(
        "--src-encoding",
        choices=[IJ_ENC, JSON_ENC, PKL_ENC, AUTO_ENC], default="auto",
        help="source encoding. defaults to 'auto', selecting a encoding based on file extension")
    convert.add_argument(
        "-e", "--dest-encoding",
        choices=[IJ_ENC, JSON_ENC, PKL_ENC], required=True,
        help="destination encoding")

    args = root.parse_args()
    cmd = args.command
    if cmd == "convert":
        handle_convert(args)
    else:
        sys.exit(1)

def handle_convert(args):
    src = args.src
    src_encoding = args.src_encoding
    dest = args.dest
    dest_encoding = args.dest_encoding
    roi_path: pathlib.Path = dest / "rois.zip"
    if roi_path.is_file():
        if not confirm_overwrite(roi_path):
            sys.exit(0)
        else:
            os.remove(roi_path)
    for s in src:
        core.convert(s, roi_path, src_encoding, dest_encoding)
