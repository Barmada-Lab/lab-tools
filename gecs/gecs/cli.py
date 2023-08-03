import argparse

from pathlib import Path

from improc.experiment.types import Channel

def parse_args():
    root_parser = argparse.ArgumentParser()

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    cvat_deploy = subparsers.add_parser('cvat-deploy')
    cvat_deploy.add_argument('project_name')
    cvat_deploy.add_argument('--ts', action='store_true', default=False)
    cvat_deploy.add_argument('images', nargs='+', type=Path)

    composite = subparsers.add_parser('composite')
    composite.add_argument('experiment_dir', type=Path)
    composite.add_argument('scratch_dir', type=Path)
    composite.add_argument('--icc-hack', action='store_true')
    composite.add_argument('--ignore', nargs='+', default=[], type=Channel)

    sns = subparsers.add_parser('sns', help='stitch n stack')
    sns.add_argument('experiment_dir', type=Path)
    sns.add_argument('scratch_dir', type=Path)
    sns.add_argument('--legacy', action='store_true', default=False)
    sns.add_argument('--out-range', default='uint16')
    sns.add_argument('--no-stitch', action='store_true', default=False)

    return root_parser.parse_args()


def main():
    args = parse_args()

    match args.command:
        case 'cvat-deploy':
            from gecs.cvat import deploy_ts, deploy_frames
            if args.ts:
                deploy_ts(args.project_name, args.images)
            else:
                deploy_frames(args.project_name, args.images)
        case 'composite':
            from gecs.composite import composite_icc_hack, composite_survival
            if args.icc_hack:                
                composite_icc_hack(args.experiment_dir, args.scratch_dir)
            else:
                composite_survival(args.experiment_dir, args.scratch_dir, args.ignore)
        case 'sns':
            from gecs.sns import stitch_n_stack
            stitch_n_stack(args.experiment_dir, args.scratch_dir, args.legacy, args.out_range, not args.no_stitch)