import argparse
from pathlib import Path

from . import core

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=Path)
    parser.add_argument("scratch_path", type=Path)
    parser.add_argument("--no-gedi", help="don't use rfp gedi", default=True, action='store_false')
    parser.add_argument("--single-cell", help="use single cell tracking", default=False, action='store_true')
    parser.add_argument("--save-stacks", help="save labelled stacks", default=False, action='store_true')
    parser.add_argument("--save-masks", help="save labelled stacks", default=False, action='store_true')
    parser.add_argument("--avg-reg", help="average registration across plate", default=False, action='store_true')
    parser.add_argument("--cpus", help="num cpus", type=int, default=1, action="store")

    return parser.parse_args()

def run():
    args = parse_args()
    core.process(args.experiment_path, args.scratch_path, args.save_stacks, args.save_masks, args.single_cell, args.no_gedi, args.avg_reg, args.cpus)
