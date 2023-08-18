from improc.experiment import Experiment, loader
from improc.processes import Pipeline, Stitch, Stack

def prep_masa(experiment: Experiment, collection: str = "raw_imgs"):
    pipeline = Pipeline(
        Stitch(),
        Stack(register=False)
    )
    pipeline.run(experiment, collection)

def cli_entry(args):
    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(args.experiment_dir, scratch_dir)
    prep_masa(experiment, args.collection)