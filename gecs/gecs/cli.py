import argparse

from pathlib import Path

def parse_args():
    root_parser = argparse.ArgumentParser()

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    cvat_deploy = subparsers.add_parser('cvat-deploy')
    cvat_deploy.add_argument('project_name')
    cvat_deploy.add_argument('img_dims')
    cvat_deploy.add_argument('images', nargs='+')

    composite = subparsers.add_parser('composite')
    composite.add_argument('experiment_dir', type=Path)
    composite.add_argument('scratch_dir', type=Path)
    composite.add_argument('--icc-hack', action='store_true')

    sns = subparsers.add_parser('sns', help='stitch n stack')
    sns.add_argument('experiment_dir', type=Path)
    sns.add_argument('scratch_dir', type=Path)
    sns.add_argument('--legacy', action='store_true', default=False)

    return root_parser.parse_args()


def main():
    args = parse_args()

    match args.command:
        case 'cvat-deploy':
            from gecs.cvat import deploy
            deploy(args.project_name, args.img_dims, args.images)
        case 'composite':
            from gecs.conversions import composite_icc_hack, composite_survival
            if args.icc_hack:                
                composite_icc_hack(args.experiment_dir, args.scratch_dir)
            else:
                composite_survival(args.experiment_dir, args.scratch_dir)
        case 'sns':
            from gecs.sns import stitch_n_stack
            stitch_n_stack(args.experiment_dir, args.scratch_dir, args.legacy)