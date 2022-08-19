from pathlib import Path
from ilmatic.util import snapshot, get_new
from improc import mobility

import shutil
import subprocess
import tifffile

def run_ilastik(ilastik_bin: Path, project: Path, *args: str):
    cmd = f"{ilastik_bin} --headless --project={project}".split(" ") + list(args)
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
        f'--output_filename_format={output}'
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

    output = output_base / "{nickname}.h5"
    args = [
        '--output_format=hdf5',
        f'--output_filename_format={output}',
        "--raw_data",
        *images,
        "--prediction_maps",
        *pix_preds,
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

    output = (output_base / "{nickname}.h5").__str__()
    args = [
        '--output_format=hdf5',
        f"--output_filename_format={output}",
        "--raw_data",
        *images,
        "--prediction_maps",
        *pix_preds,
    ]

    run_ilastik(
        ilastik_bin,
        classifier_path,
        *args)

    final = snapshot(output_base.glob("*.h5"))
    return get_new(initial, final)

def run_survival_pipeline(experiment_path: Path, classifier_path: Path, ilastik_bin):
    pixel_classifier = experiment_path / "pixel_classifier.ilp"
    object_classifier = experiment_path / "object_classifier.ilp"
    tracker = experiment_path / "tracker.ilp"

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

    pixel_probabilities = run_pixel_classifier(
        ilastik_bin,
        pixel_classifier,
        raw_imgs,
        pixel_probabilities_base)

    run_object_classifier(
        ilastik_bin,
        object_classifier,
        raw_imgs,
        pixel_probabilities,
        object_probabilities_base)

    run_tracker(
        ilastik_bin,
        tracker,
        raw_imgs,
        pixel_probabilities,
        tracking_results_base)

def run_mito_pipeline(experiment_path: Path, classifier_path: Path, scratch: Path, ilastik_bin):
    motion_classifier = classifier_path / "mito" / "mobility_pixel_classifier.ilp"
    still_classifier = classifier_path / "mito" / "still_pixel_classifier.ilp"
    stacks = experiment_path / "processed_imgs" / "stacked"

    if not motion_classifier.exists():
        raise FileNotFoundError("No motion classifier found")
    elif not still_classifier.exists():
        raise FileNotFoundError("No still classifier found")
    elif not experiment_path.exists():
        raise FileNotFoundError("Experiment path does not exist")
    elif not stacks.exists():
        raise FileNotFoundError("No stacks found")
    elif not scratch.exists():
        raise FileNotFoundError("Scratch path does not exist")

    orig = list(stacks.glob("*.tif"))
    hilight_output_dir = scratch / "hilighted"
    if not hilight_output_dir.exists():
        hilight_output_dir.mkdir()

    still_outputs_dir = scratch / "still_trimmed"
    if not still_outputs_dir.exists():
        still_outputs_dir.mkdir()

    pairs = []
    for stack in orig:
        print(stack)
        try:
            img = tifffile.imread(stack)
            motion_output_path = hilight_output_dir / stack.name
            hilighted = mobility.hilight_motion(img)
            tifffile.imwrite(motion_output_path, hilighted)
            still_output_path = still_outputs_dir / stack.name
            trimmed = img[0]
            tifffile.imwrite(still_output_path, trimmed)
            pairs.append((still_output_path, motion_output_path))
        except:
            continue

    motion_outputs = scratch / "motion"
    if not motion_outputs.exists():
        motion_outputs.mkdir()

    still_outputs = scratch / "still"
    if not still_outputs.exists():
        still_outputs.mkdir()


    still_imgs = list(zip(*pairs))[0]
    run_pixel_classifier(ilastik_bin, still_classifier, list(still_imgs), still_outputs, axes="yx")

    motion_imgs = list(zip(*pairs))[1]
    run_pixel_classifier(ilastik_bin, motion_classifier, list(motion_imgs), motion_outputs)

    return (list(still_outputs.glob("*")), list(motion_outputs.glob("*")))
