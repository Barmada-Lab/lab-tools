from pathlib import Path

from trogon import tui
import click

from lab_tools.analysis.tasks import run_pultra_survival
from lab_tools.utils import experiment_path_argument, experiment_type_argument
from lab_tools.experiment import ExperimentType


@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@click.command("pultra-survival")
@experiment_path_argument()
@experiment_type_argument()
@click.option("--save-annotations", is_flag=True,
              help="Save annotated stacks to results folder")
@click.option("--sync", is_flag=True,
              help="Run synchronously, skipping the task queue. Useful for debugging.")
def pultra_survival(experiment_path: Path, experiment_type: ExperimentType, save_annotations: bool, sync: bool):

    if sync:
        from lab_tools.analysis.pultra_survival import run
        run(experiment_path, experiment_type, save_annotations)
    else:
        run_pultra_survival.delay(str(experiment_path), experiment_type, save_annotations)


cli.add_command(pultra_survival)
