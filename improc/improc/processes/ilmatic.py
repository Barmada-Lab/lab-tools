from pathlib import Path

from dataclasses import dataclass
import pathlib
import subprocess
from typing import Iterable

@dataclass(eq=True, frozen=True)
class FileSnapshot:
    path: pathlib.Path
    modified_time: int

def snapshot(paths: Iterable[pathlib.Path]) -> set[FileSnapshot]:
    return set([FileSnapshot(path, path.stat().st_mtime_ns) for path in paths])

def get_new(before: set[FileSnapshot], after: set[FileSnapshot]) -> list[pathlib.Path]:
    return sorted(snap.path for snap in after - before)

def run_ilastik(ilastik_bin: Path, project: Path, *args: str):
    cmd = [
        ilastik_bin,
        "--headless",
        "--readonly",
        "--project", project,
    ] + list(args)
    proc = subprocess.run(cmd)
    proc.check_returncode()


def run_pixel_classifier(ilastik_bin, classifier_path: Path, images: list[Path], output_base: Path, axes="tyx"):
    if not classifier_path.exists():
        raise FileNotFoundError("Pixel classifier does not exist")

    initial = snapshot(output_base.glob("*.h5"))

    output = output_base / "{nickname}.h5"
    args = [
        '--input_axes', axes,
        '--output_format', 'hdf5',
        '--output_filename_format', output,
        '--export_source', 'probabilities',
    ] + list(map(str, images))

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

    args = [
        '--output_format', 'hdf5',
        '--output_filename_format', output_base / "{nickname}.h5",
        '--export_source', 'object probabilities',
        "--raw_data", *images,
        "--prediction_maps", *pix_preds,
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

    args = [
        '--output_format', 'hdf5',
        "--output_filename_format", output_base / "{nickname}.h5",
        "--raw_data", *images,
        "--prediction_maps", *pix_preds,
    ]

    run_ilastik(
        ilastik_bin,
        classifier_path,
        *args)

    final = snapshot(output_base.glob("*.h5"))
    return get_new(initial, final)

def run_survival_pipeline(experiment_path: Path, classifier_path: Path, ilastik_bin, run_prob=True, run_obj=True, track=True):
    pixel_classifier = classifier_path / "pixel_classifier.ilp"
    object_classifier = classifier_path / "object_classifier.ilp"
    tracker = classifier_path / "tracker.ilp"

    if not pixel_classifier.exists():
        raise FileNotFoundError(f"No pixel classifier at path {pixel_classifier}")
    elif not object_classifier.exists():
        raise FileNotFoundError(f"No object classifier at path {object_classifier}")
    elif not tracker.exists():
        raise FileNotFoundError(f"No tracker at path {tracker}")

    pixel_probabilities_base = experiment_path / "processed_imgs" / "pixel_probabilities"
    object_probabilities_base = experiment_path / "processed_imgs" / "object_probabilities"
    tracking_results_base = experiment_path / "processed_imgs" / "tracking_results"

    raw_imgs = sorted((experiment_path / "processed_imgs" / "stacked").glob("*.tif"))

    if run_prob:
        pixel_probabilities = run_pixel_classifier(
            ilastik_bin,
            pixel_classifier,
            raw_imgs,
            pixel_probabilities_base)
    else:
        pixel_probabilities = sorted(pixel_probabilities_base.glob("*.h5"))

    if run_obj:
        run_object_classifier(
            ilastik_bin,
            object_classifier,
            raw_imgs,
            pixel_probabilities,
            object_probabilities_base)

    if track:
        run_tracker(
            ilastik_bin,
            tracker,
            raw_imgs,
            pixel_probabilities,
            tracking_results_base)
