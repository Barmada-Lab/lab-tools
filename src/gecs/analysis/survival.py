import pathlib as pl
import shutil
import csv
from typing import Any, Generator
from contextlib import contextmanager
from itertools import product

import click
from dask.distributed import Client, wait
from lifelines import CoxPHFitter, KaplanMeierFitter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from dask_jobqueue import SLURMCluster
import matplotlib.pyplot as plt
import xarray as xr
import tqdm
from PIL import Image

from gecs.io.lux_loader import read_lux_experiment
from gecs.io.legacy_loader import read_legacy_experiment
from gecs.display import stitch, illumination_correction, clahe, rescale_intensity
from gecs.segmentation import annotate_segmentation, segment_clahe

def slurm_cluster(scratch: pl.Path):
    @contextmanager
    def _slurm_cluster() -> Generator[Client, Any, None]:
        cluster = SLURMCluster(
            account="sbarmada0",
            walltime="01:00:00",
            cores=20,
            memory="90 GB",
            interface="ib0",
            local_directory=scratch,
            scheduler_options= dict(
                dashboard_address=f"0.0.0.0:0"
            ),
            worker_extra_args=["--lifetime", "50m", "--lifetime-stagger", "4m"],
        )
        cluster.scale(jobs=1)
        client = Client(cluster)
        try:
            yield client
        finally:
            client.shutdown()
    return _slurm_cluster

def write_survival_results(output_dir: pl.Path, labeled: xr.DataArray, well_csv: pl.Path | None = None):
    count_rows = []
    for well, field in product(labeled.well, labeled.field):
        for t in labeled.t:
            frame = labeled.sel(well=well, field=field, t=t)
            count_rows.append({
                "well": str(well.values),
                "field": str(field.values),
                "time": int(t.values),
                "count": len(np.unique(frame.values))
            })
    
    cellcounts = pd.DataFrame.from_records(count_rows)
    cellcount_output_path = output_dir / "raw_cell_counts.csv"
    cellcounts.to_csv(cellcount_output_path, index=False)

    counts = cellcounts.groupby(["well", "time"]).sum()
    death_rows = []
    for well in cellcounts.well.unique():
        trend = counts.loc[well,]["count"] #type: ignore
        smoothed = gaussian_filter1d(trend, sigma=1.5)
        diff = np.diff(smoothed)
        for idx, deaths in enumerate(diff):
            if deaths > 0:
                for _ in range(deaths):
                    death_rows.append({
                        "well": well,
                        "time": idx + 2,
                        "dead": 1
                    })
        for _ in range(smoothed[-1]):
            # add censored events for cells that survived during observation
            death_rows.append({
                "well": well,
                "time": len(smoothed) + 1,
                "dead": 0
            })

    deaths = pd.DataFrame.from_records(death_rows)
    deaths_output_path = output_dir / "smoothed_deaths.csv"
    deaths.to_csv(deaths_output_path, index=False)

    if well_csv is not None:
        wells = pd.read_csv(well_csv)
        if "Condition" not in wells.columns:
            return

        wells["well"] = wells["Vertex"].str.replace("well_", "")
        wells = wells[["well", "Condition"]]
        merged = deaths.merge(wells, on="well").drop(columns="well")
        cph = CoxPHFitter()
        cph.fit(merged, duration_col="time", event_col="dead", strata="Condition")
        cph.plot()
        cph.baseline_cumulative_hazard_.plot(
            ylabel="Cumulative hazard",
            xlabel="T",
            title="Baseline cumulative hazards",
            drawstyle="steps")
        output_fig = output_dir / "CoxPH_baselines.pdf"
        plt.savefig(output_fig, format="pdf")

def write_annotations(annotated_path: pl.Path, annotated: xr.DataArray, client):
    annotated_path.mkdir(exist_ok=True)

    def _write_ts_as_gifs(stack, well):
        path = annotated_path / f"{well}.gif"
        data = np.array(stack)
        frame_0 = Image.fromarray(data[0])
        the_rest = [Image.fromarray(frame) for frame in data[1:]]
        frame_0.save(path, format='GIF', save_all=True, 
            append_images=the_rest, duration=500, loop=0)
        
    futures = []
    for well in annotated.well:
        stack = annotated.sel(well=well)
        futures.append(
            client.submit(_write_ts_as_gifs, stack, well.values))
    wait(futures)

def local_cluster():
    @contextmanager
    def _local_cluster() -> Generator[Client, Any, None]:
        client = Client()
        try:
            yield client
        finally:
            client.shutdown()
    return _local_cluster

def gpu_cluster():
    @contextmanager
    def _gpu_cluster() -> Generator[Client, Any, None]:
        client = Client(processes=False)
        try:
            yield client
        finally:
            client.shutdown()
    return _gpu_cluster

def gfp_method(
        experiment: xr.Dataset, 
        gpu_cluster, 
        cpu_cluster, 
        model_loc, 
        scratch_dir,
        output_dir,
        well_csv: pl.Path | None = None):

    intensity = experiment.intensity.sel(channel="GFP")

    if not scratch_dir.exists():
        with gpu_cluster() as client:

            print(f"dashboard: {client.dashboard_link}")

            corrected = illumination_correction(intensity, ["t","y","x"])
            clahed = clahe(corrected)
            rescaled = rescale_intensity(clahed, ["y","x"], out_range=np.float64)
            labeled = segment_clahe(rescaled, str(model_loc))
            try:
                xr.Dataset({
                    "labeled": labeled,
                }).to_zarr(scratch_dir, mode="w")
            except Exception as e:
                print("Encountered an error while writing to zarr: ", e)
                shutil.rmtree(scratch_dir)
                return
    
    with cpu_cluster() as client:

        data = xr.open_zarr(scratch_dir)
        labeled = data.labeled
        write_survival_results(output_dir, labeled, well_csv)

        corrected = illumination_correction(intensity, ["t","y","x"]).persist()
        annotated = annotate_segmentation(stitch(corrected), stitch(labeled))
        write_annotations(output_dir / "annotated", annotated, client)

@click.command("survival")
@click.argument("experiment_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.argument("scratch", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
@click.option("--legacy", is_flag=True, default=False)
@click.option("--use-slurm", is_flag=True, default=False)
@click.option("--mode", type=click.Choice(['gfp', 'gedi'], case_sensitive=False))
@click.option("--model-loc", type=click.Path(exists=True, path_type=pl.Path))
def cli_entry(
        experiment_base: pl.Path, 
        scratch: pl.Path, 
        legacy: bool,
        use_slurm: bool, 
        mode: str,
        model_loc: pl.Path):

    experiment = read_legacy_experiment(experiment_base) if legacy else read_lux_experiment(experiment_base)
    if (experiment_base / "wells.csv").exists():
        well_csv = experiment_base / "wells.csv"
    else:
        well_csv = None
    
    cluster_handle = slurm_cluster(scratch) if use_slurm else local_cluster()

    scratch_dir = scratch / f"{experiment_base.name}.zarr"
    output_dir = experiment_base / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "gfp":
        return gfp_method(experiment, gpu_cluster(), cluster_handle, model_loc, scratch_dir, output_dir, well_csv)