from pathlib import Path

from trogon import tui
import click

from cytomancer.utils import test_cvat_connection
from cytomancer.experiment import ExperimentType
from cytomancer.cvat.survival import cli_entry as cvat_survival
from cytomancer.cvat.upload import cli_entry_experiment
from cytomancer.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

from cytomancer.updater import check_for_updates
from cytomancer.config import config


def experiment_dir_argument(**kwargs):
    return click.argument(
        "experiment_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        **kwargs)


def experiment_type_argument(**kwargs):
    return click.argument(
        "experiment_type",
        type=click.Choice(ExperimentType.__members__),  # type: ignore
        callback=lambda c, p, v: getattr(ExperimentType, v) if v else None,
        **kwargs)


@tui()
@click.group(help="Cytomancer CLI")
@click.pass_context
def cli(ctx):
    check_for_updates()
    ctx.ensure_object(dict)


# ----------------- BEGIN QUANT GROUP -----------------

@click.command("pult-surv")
@experiment_dir_argument()
@click.option("--classifier-name", default="nuclei_survival_svm.joblib", show_default=True, help="Name of pretrained StarDist model to use.")
@click.option("--save-annotations", is_flag=True, help="Save annotated stacks to results folder")
@click.option("--run-sync", is_flag=True, help="Run synchronously, skipping the task queue.")
def pultra_survival(experiment_dir: Path, classifier_name, save_annotations: bool, run_sync: bool):
    """
    Run pultra survival analysis on an experiment. Note that only CQ1 acquisitions are supported.
    """
    svm_path = config.models_dir / classifier_name

    if run_sync:
        from cytomancer.quant.pultra_survival import run
        from dask.distributed import LocalCluster, Client
        _ = Client(LocalCluster(n_workers=8, threads_per_worker=2))
        run(experiment_dir, ExperimentType.CQ1, svm_path, save_annotations)
    else:
        from cytomancer.quant.tasks import run_pultra_survival
        run_pultra_survival.delay(str(experiment_dir), ExperimentType.CQ1, str(svm_path), save_annotations)


@cli.group("quant", help="Tools for quantifying data")
@click.pass_context
def quant_group(ctx):
    ctx.ensure_object(dict)


quant_group.add_command(pultra_survival)


# ----------------- BEGIN CVAT GROUP -----------------

@cli.group("cvat", help="Tools for working with CVAT")
@click.pass_context
def cvat_group(ctx):
    ctx.ensure_object(dict)


@click.command("auth")
@click.option("--cvat-username", prompt="CVAT Username")
@click.password_option("--cvat-password", prompt="CVAT Password", confirmation_prompt=False)
def cvat_auth(cvat_username, cvat_password):
    """
    Update CVAT credentials. Run this with no arguments to get an interactive prompt that hides your password.
    """

    print(f"\nTesting CVAT connection to server {config.cvat_url}...")
    if not test_cvat_connection(config.cvat_url, cvat_username, cvat_password):
        print("Connection failed. Please verify your credentials and try again.")
        print("See `cyto config update --help` for other CVAT-related settings")
        return

    print("Authentication successful. Saving credentials.")
    config.cvat_username = cvat_username
    config.cvat_password = cvat_password
    config.save()


cvat_group.add_command(cvat_auth)
cvat_group.add_command(cvat_survival)
cvat_group.add_command(cli_entry_experiment)
cvat_group.add_command(cvat_nuc_cyto)


# ----------------- BEGIN SETTINGS GROUP -----------------

@cli.group("config", help="Config management")
@click.pass_context
def config_group(ctx):
    ctx.ensure_object(dict)


@click.command("show")
def show_config():
    """
    Display current configuration settings.
    """
    print("\nCurrent settings:")
    for k, v in config.model_dump().items():
        if k == "cvat_password":
            v = "*" * len(v)
        print(f"\t{k}: {v}")
    print()


# Hacky little thing that adds options for all settings to set_config
def settings_options():
    def combined_decorator(func):
        for k, v in reversed(config.model_dump().items()):
            if k == "cvat_password" or k == "cvat_username":
                continue
            decorator = click.option(f"--{k}", default=v, show_default=True)
            func = decorator(func)
        return func
    return combined_decorator


@click.command("update")
@settings_options()
def update_config(**kwargs):
    """
    Update config
    """
    for k, v in kwargs.items():
        setattr(config, k, v)
    config.save()


config_group.add_command(show_config)
config_group.add_command(update_config)

# ----------------- BEGIN ONEOFFS GROUP -----------------


@cli.group("oneoffs", help="A place for scripts that I wrote once and might be useful again")
@click.pass_context
def oneoffs_group(ctx):
    ctx.ensure_object(dict)


@click.command("stardist-seg-cvat-proj")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--channel", "-c", default="DAPI", show_default=True, help="Channel to segment.")
@click.option("--label-name", "-l", default="dead", show_default=True, help="Name of the label to create.")
@click.option("--adapteq-clip-limit", "-c", default=0.01, show_default=True, help="Clip limit for adaptive histogram equalization.")
@click.option("--median-filter-d", "-m", default=5, show_default=True, help="Diameter of the median filter to apply to the images before segmentation.")
@click.option("--model-name", "-m", default="2D_versatile_fluo", show_default=True, help="Name of predefined StarDist model to use.")
def stardist_seg_cvat_proj(
        project_name: str,
        experiment_dir: Path,
        experiment_type: ExperimentType,
        channel: str,
        label_name: str,
        adapteq_clip_limit: float,
        median_filter_d: int,
        model_name: str):

    """ Segment images in a CVAT project using StarDist. """
    from cytomancer.oneoffs.tasks import stardist_seg_cvat_proj_run
    stardist_seg_cvat_proj_run.delay(
        project_name,
        experiment_dir,
        experiment_type,
        channel,
        label_name,
        adapteq_clip_limit,
        median_filter_d,
        model_name)


oneoffs_group.add_command(stardist_seg_cvat_proj)
