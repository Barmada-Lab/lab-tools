from improc.experiment import Experiment, loader
from improc.processes import Pipeline, BaSiC

def flatfield(experiment: Experiment, collection: str, group_by: list[str], parallelism: int = 4):
    pipeline = Pipeline(
        BaSiC(group_by=group_by, parallelism=parallelism))
    pipeline.run(experiment, collection)

def cli_entry(args):
    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(args.experiment_dir, scratch_dir)
    flatfield(experiment, args.collection, args.group_by, args.parallelism)