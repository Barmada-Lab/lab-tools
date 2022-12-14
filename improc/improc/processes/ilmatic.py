import os
import csv
import subprocess
from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py
from sklearn.preprocessing import normalize

from improc.common.result import Result, Value
from improc.experiment.types import Dataset, Experiment
from improc.processes.types import Task, TaskError

@dataclass(eq=True, frozen=True)
class FileSnapshot:
    path: Path
    modified_time: int

def snapshot(paths: Iterable[Path]) -> set[FileSnapshot]:
    return set([FileSnapshot(path, path.stat().st_mtime_ns) for path in paths])

def get_new(before: set[FileSnapshot], after: set[FileSnapshot]) -> list[Path]:
    return sorted(snap.path for snap in after - before)

def run_ilastik(ilastik_bin: Path, project: Path, *args: str):
    cmd = [
        ilastik_bin,
        "--headless",
        f"--project={project}",
    ] + list(args)
    proc = subprocess.run(cmd)
    proc.check_returncode()


def run_pixel_classifier(ilastik_bin, classifier_path: Path, images: list[Path], output_base: Path, axes="tyx"):
    if not classifier_path.exists():
        raise FileNotFoundError("Pixel classifier does not exist")

    initial = snapshot(output_base.glob("*.h5"))

    output = output_base / "{nickname}.h5"
    args = [
        '--output_format=hdf5',
        f'--input_axes={axes}',
        f'--output_filename_format={output}',
        "--raw_data",
        *list(map(str, images))
    ] 

    run_ilastik(
        ilastik_bin,
        classifier_path,
        *args)

    final = snapshot(output_base.glob("*.h5"))
    return get_new(initial, final)

def run_object_classifier(ilastik_bin, classifier_path: Path, images: list[Path], pix_preds: list[Path], output_base: Path):
    if not classifier_path.exists():
        raise FileNotFoundError("Object classifier does not exist")

    initial = snapshot(output_base.glob("*.h5"))

    output = output_base / "{nickname}.h5"
    args = [
        '--output_format=hdf5',
        f'--output_filename_format={output}',
        '--output_source="Object Probabilities"',
        "--raw_data",
        *list(map(str,images)),
        "--prediction_maps", 
        *list(map(str,pix_preds)),
    ]

    run_ilastik(
        ilastik_bin,
        classifier_path,
        *args)

    final = snapshot(output_base.glob("*.h5"))
    return get_new(initial, final)

def run_tracker(ilastik_bin, classifier_path: Path, images: list[Path], pix_preds: list[Path], output_base: Path):
    if not classifier_path.exists():
        raise FileNotFoundError("Tracker does not exist")

    initial = snapshot(output_base.glob("*.h5"))

    output = output_base / "{nickname}.h5",
    args = [
        '--output_format=hdf5',
        f"--output_filename_format={output}", 
        "--raw_data", 
        *list(map(str, images)),
        "--prediction_maps", 
        *list(map(str,pix_preds)),
    ]

    run_ilastik(
        ilastik_bin,
        classifier_path,
        *args)

    final = snapshot(output_base.glob("*.h5"))
    return get_new(initial, final)

def run_survival_pipeline(image_path: Path, classifier_path: Path, ilastik_bin, run_prob=True, run_obj=True, track=True):
    pixel_classifier = classifier_path / "pixel_classifier.ilp"
    object_classifier = classifier_path / "object_classifier.ilp"
    tracker = classifier_path / "tracker.ilp"

    if not pixel_classifier.exists():
        raise FileNotFoundError(f"No pixel classifier at path {pixel_classifier}")
    elif not object_classifier.exists():
        raise FileNotFoundError(f"No object classifier at path {object_classifier}")
    elif not tracker.exists():
        raise FileNotFoundError(f"No tracker at path {tracker}")

    pixel_probabilities_base = image_path / "pixel_probabilities"
    object_probabilities_base = image_path / "object_probabilities"
    tracking_results_base = image_path / "tracking_results"

    raw_imgs = sorted((image_path / "stacked").glob("*.tif"))

    if run_prob:
        pixel_probabilities = run_pixel_classifier(
            ilastik_bin,
            pixel_classifier,
            raw_imgs,
            pixel_probabilities_base)
    else:
        pixel_probabilities = sorted(pixel_probabilities_base.glob("*.h5"))

    if run_obj:
        object_probabilities = run_object_classifier(
            ilastik_bin,
            object_classifier,
            raw_imgs,
            pixel_probabilities,
            object_probabilities_base)

        paired = []
        for pixel_prob in pixel_probabilities:
            for object_prob in object_probabilities:
                if pixel_prob.name == object_prob.name:
                    paired.append((pixel_prob, object_prob))

        for pixel_prob, object_prob in paired:
            with h5py.File(pixel_prob, "r+") as pix, h5py.File(object_prob) as o:
                live = o["exported_data"][:,:,:,0] # type: ignore
                live[live != 1] = 0 #type: ignore
                pixel_map = pix["exported_data"][...] #type: ignore
                pixel_map[:,:,:,2] *= live #type: ignore
                normalized = normalize(pixel_map.reshape(-1, 3)).reshape(pixel_map.shape) #type: ignore
                del pix["exported_data"]
                pix.create_dataset("exported_data", data=normalized, chunks=True)

    if track:
        run_tracker(
            ilastik_bin,
            tracker,
            raw_imgs,
            pixel_probabilities,
            tracking_results_base)


class SurvivalAnalysis(Task):

    def __init__(self, classifier_path, ilastik_bin, death_threshold= 0.5) -> None:
        super().__init__("survival_pipeline")
        self.classifier_path = classifier_path
        self.ilastik_bin = ilastik_bin
        self.death_threshold = death_threshold

    def _track(self, labels_arr, tracking_arr):
        lines = {k:[] for k in np.unique(tracking_arr[0]) if k > 1}
        ctr = 0
        for labels, tracking in zip(labels_arr, tracking_arr):
            tp = {}
            for _idx in np.argwhere(tracking > 1):
                idx = tuple(_idx)
                id = tracking[idx]
                if id in tp: # skip if we've already recorded this id in this timepoint
                    continue
                label = labels[idx]
                if label > 0:
                    tp[id] = label

            for id, label in tp.items():
                if len(lines[id]) == ctr:
                    lines[id].append(label)

            ctr += 1
        return { k: np.array(v) for k,v in lines.items() }

    def _get_wellspec(self, label, wells):
        for wellspec in wells:
            if wellspec.label == label:
                return wellspec
        return None

    def _format_row(self, well, line, line_id, tps, death_threshold):
        censored = False
        # only look at live cells
        if line[0] < death_threshold:
            return None

        # observed death
        dead_at = -1
        for idx, label in enumerate(line):
            if label > death_threshold:
                dead_at = idx
                break

        # unobserved death (segmentation fails)
        if len(line) < tps and dead_at == -1:
            dead_at = len(line) - 1

        death_cause = 'death' if dead_at != -1 else 'NA'
        event = 1
        censored = 0 if censored else 1 # yea i know
        last_tp = tps if dead_at == -1 else dead_at + 1 # 1-indexed
        last_time = last_tp

        group = "-".join(drug.drug_label for drug in well.drugs)
        if group == "":
            group = "NA"

        row = {
            "well": well.label, # well...
            "id": line_id,
            "well-id": f"{well.label}-{line_id}",
            "group": group,
            "cell_type": "NA",
            "drug": "NA",
            "drug_conc": "NA",
            "column": "NA",
            "last_tp": last_tp,
            "last_time": last_time,
            "death_cause": death_cause,
            "censored": censored,
            "event": event
        }

        return row

    def _ilastik_to_barma(self, experiment: Experiment):
        fieldnames = ['well','id','well-id','group','cell_type','drug','drug_conc','column','last_tp','last_time','death_cause','censored','event']
        experiment_base = experiment.experiment_dir
        output_dir = experiment_base / "results"
        os.makedirs(output_dir, exist_ok=True)

        label_paths = list((experiment_base / "processed_imgs" / "object_predictions").glob("*.h5"))
        track_paths = list((experiment_base / "processed_imgs" / "tracking").glob("*.h5"))

        with open(output_dir / "survival_data.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            label_track_paired = []
            for label_e in label_paths:
                for track_e in track_paths:
                    label_vertex = label_e.name
                    track_vertex = track_e.name
                    if label_vertex == track_vertex:
                        label_track_paired.append((label_e, track_e))

            for label_path, track_path in label_track_paired:
                labels = h5py.File(label_path)["exported_data"]
                tracks = h5py.File(track_path)["exported_data"]
                tracking_lines = self._track(labels, tracks)
                well_label = label_path.name.split("-")[0]
                well = self._get_wellspec(well_label, experiment.mfspec)
                tps = max(map(len, tracking_lines.values()))
                rows = [self._format_row(well, line, idx, tps, self.death_threshold) for idx, line in enumerate(tracking_lines.values())]
                rows = [row for row in rows if row is not None]
                writer.writerows(rows)

    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        input_path = experiment.scratch_dir
        run_survival_pipeline(input_path, self.classifier_path, self.ilastik_bin)
        self._ilastik_to_barma(experiment)
        return Value(dataset)
