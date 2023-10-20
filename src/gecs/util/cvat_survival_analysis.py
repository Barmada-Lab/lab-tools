from typing import Any
import pathlib as pl

from lifelines import CoxPHFitter
import click
import matplotlib.pyplot as plt
from cvat_sdk import make_client, Client
from skimage.measure import regionprops
import pandas as pd
from tqdm import tqdm

from ..settings import settings
from .cvat_upload_experiment import prep_experiment

def extract_survival_result(track) -> dict[str, Any]:
    points = track.shapes
    for point in points:
        if point.outside:
            return {
                "time": point.frame,
                "dead": 1
            }
    return {
        "time": points[-1].frame,
        "dead": 0
    }

def analyze_survival(
        client: Client,
        project_id: int,
        output_dir: pl.Path,
        well_csv: pl.Path|None):

    tasks = client.projects.retrieve(project_id).get_tasks()
    rows = []
    for task_meta in tqdm(tasks):
        well = task_meta.name[:3]
        annotation = task_meta.get_annotations()
        for track in annotation.tracks:
            survival_result = extract_survival_result(track)
            survival_result["well"] = well
            rows.append(survival_result)
    df = pd.DataFrame.from_records(rows)

    if well_csv is not None:
        well_df = pd.read_csv(well_csv)[["Vertex", "Condition"]]
        well_df["well"] = well_df["Vertex"].str[-3:]
        well_df = well_df.drop(columns=["Vertex"])
        df = df.merge(well_df, on="well")
        cph = CoxPHFitter()
        cph.fit(df.drop(columns="well"), duration_col="time", event_col="dead", strata="Condition")
        cph.baseline_cumulative_hazard_.plot(
            ylabel="Cumulative hazard",
            xlabel="T",
            title="Baseline cumulative hazards",
            drawstyle="steps")
        output_fig = output_dir / "CoxPH_baselines.pdf"
        plt.savefig(output_fig, format="pdf")

    output_csv = output_dir / "survival.csv"
    df.to_csv(output_csv, index=False)

@click.command("cvat-survival-analysis")
@click.argument("project_name")
@click.argument("output_dir", type=click.Path(path_type=pl.Path))
@click.option("--well-csv", type=click.Path(path_type=pl.Path), default=None)
def cli_entry(project_name: str, output_dir: pl.Path, well_csv: pl.Path|None):
    with make_client(
        host=settings.cvat_url,
        credentials=(
            settings.cvat_username,
            settings.cvat_password
        )
    ) as client:
        org_slug = settings.cvat_org_slug
        client.organization_slug = org_slug
        api_client = client.api_client

        (data, _) = api_client.projects_api.list(search=project_name)
        assert data is not None and len(data.results) > 0, \
            f"No project matching {project_name} in organization {org_slug}"

        try:
            # exact matches only
            project = next(filter(lambda x: x.name == project_name, data.results))
        except StopIteration:
            raise ValueError(f"No project matching {project_name} in organization {org_slug}")

        output_dir.mkdir(exist_ok=True)

        project_id = project.id
        analyze_survival(client, project_id, output_dir, well_csv)