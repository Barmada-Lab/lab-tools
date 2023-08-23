import argparse

from pathlib import Path

from improc.experiment.types import Channel

from . import composite, correlate, cvat_deploy, flatfield, measure, sns, masa_prep

def parse_args():
    root_parser = argparse.ArgumentParser()
    root_parser.add_argument('--parallelism', type=int, help='Number of parallel processes', default=1)

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    composite_parser = subparsers.add_parser('composite')
    composite_parser.add_argument('experiment_dir', type=Path)
    composite_parser.add_argument('--scratch_dir', type=Path, default=None)
    composite_parser.add_argument('--icc-hack', action='store_true')
    composite_parser.add_argument('--ignore', nargs='+', default=[], type=Channel)
    composite_parser.set_defaults(func=composite.cli_entry)

    sns_parser = subparsers.add_parser('sns', help='stitch n stack')
    sns_parser.add_argument('experiment_dir', type=Path)
    sns_parser.add_argument('--scratch_dir', type=Path, default=None)
    sns_parser.add_argument('--collection', type=str, default="raw_imgs", help='Collection name')
    sns_parser.add_argument('--legacy', action='store_true', default=False, help='Use legacy stitching')
    sns_parser.add_argument('--out-range', default='uint16', help='Output range')
    sns_parser.add_argument('--no-stitch', action='store_true', default=False)
    sns_parser.set_defaults(func=sns.cli_entry)

    flatfield_parser = subparsers.add_parser('flatfield', help='Flatfield correction, uses BaSiC')
    flatfield_parser.add_argument('experiment_dir', type=Path, help='Experiment directory')
    flatfield_parser.add_argument('--collection', type=str, default="raw_imgs", help='Collection name')
    flatfield_parser.add_argument('--scratch-dir', type=Path, default=None)
    flatfield_parser.add_argument('--group-by', nargs='+', default=['vertex', 'mosaic', 'exposure'], help='Group by')
    flatfield_parser.set_defaults(func=flatfield.cli_entry)

    measure_parser = subparsers.add_parser('measure', help='measure rois')
    measure_parser.add_argument('raw_dir', type=Path)
    measure_parser.add_argument('roi_dir', type=Path)
    measure_parser.add_argument('--output', type=Path, default=None)
    measure_parser.add_argument('--avg', action='store_true', default=False)
    measure_parser.add_argument('--median', action='store_true', default=False)
    measure_parser.add_argument('--std', action='store_true', default=False)
    measure_parser.add_argument('--area', action='store_true', default=False)
    measure_parser.add_argument('--cumhist', action='store_true', default=False)
    measure_parser.set_defaults(func=measure.cli_entry)

    correlate_parser = subparsers.add_parser('correlate', help='correlate rois')
    correlate_parser.add_argument('roi_dir1', type=Path)
    correlate_parser.add_argument('roi_dir2', type=Path)
    correlate_parser.add_argument('--output', type=Path, default=None)
    correlate_parser.set_defaults(func=correlate.cli_entry)

    masa_prep_parser = subparsers.add_parser('masa-prep')
    masa_prep_parser.add_argument('experiment_dir', type=Path)
    masa_prep_parser.add_argument('--scratch_dir', type=Path, default=None)
    masa_prep_parser.add_argument('--collection', type=str, default="raw_imgs", help='Collection name')
    masa_prep_parser.set_defaults(func=masa_prep.cli_entry)

    cvat_parser= subparsers.add_parser('cvat-deploy')
    cvat_parser.add_argument('project_name')
    cvat_parser.add_argument('--ts', action='store_true', default=False)
    cvat_parser.add_argument('images', nargs='+', type=Path)
    cvat_parser.set_defaults(func=cvat_deploy.cli_entry)

    return root_parser.parse_args()

def main():
    args = parse_args()
    args.func(args)