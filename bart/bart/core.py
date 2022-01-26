from posixpath import basename
from typing import Iterable, Tuple
import globus_sdk

from uuid import uuid4
from globus_sdk.services import transfer
from joblib import Parallel, delayed

from glob import glob
import shutil
import subprocess
import time
import sys
import os

from .helpers import DynamicStr

REFRESH_TOKEN = "refresh_token"
ACCESS_TOKEN = "access_token"

AUTH_SERVER = "auth.globus.org"
TRANSFER_SERVER = "transfer.api.globus.org"

SCRATCH_PATH = os.path.abspath("/scratch/sbarmada_root/sbarmada0/shared_data/tmp/")
TURBO_PATH = os.path.abspath("/nfs/turbo/umms-sbarmada")
ARCHIVE_PATH = os.path.abspath("/nfs/dataden/umms-sbarmada/experiments")

FLUX_UUID = "f94e0c94-f006-11e7-8219-0a208f818180"
GREATLAKES_UUID = "e0370902-9f48-11e9-821b-02b7a92d8e58"

def list_experiments(client: globus_sdk.TransferClient):
    print("Experiments:")
    print("------------")
    for entry in client.operation_ls(FLUX_UUID, path=ARCHIVE_PATH):
        name = entry["name"].replace(".tar.gz","")
        print(name)


def push_experiments(client: globus_sdk.TransferClient, *experiment_dirs):
    if not (verify_environment() and
            all(verify_experiment_directory(directory) for directory in experiment_dirs)):
        print("exiting...")
        sys.exit(1)

    clear_scratch()

    transfer = globus_sdk.TransferData(
        client,
        source_endpoint=GREATLAKES_UUID,
        destination_endpoint=FLUX_UUID,
        sync_level="checksum"
    )

    def push_experiment(experiment_path):
        tar_output = os.path.join(SCRATCH_PATH, f"{uuid4()}.tar.gz")
        print(f"compressing {experiment_path} to tarfile...")
        parent_directory = os.path.abspath(os.path.join(experiment_path, os.path.pardir))
        returncode = subprocess.run(
            ["tar", "-I", "pigz", "-cf", tar_output, os.path.basename(experiment_path)],
            cwd=parent_directory,
        ).returncode
        return (os.path.basename(experiment_path), tar_output, returncode)

    results_raw = Parallel(n_jobs=len(experiment_dirs), prefer="threads")(map(delayed(push_experiment), experiment_dirs))

    # dumb typecasting stuff
    results: Iterable[Tuple[str,str,int]] = []
    if results_raw is not None:
        results = results_raw

    for experiment_name, input_dir, returncode in results:
        if returncode != 0:
            print(f"Received a nonzero exit code while archiving {input_dir};")
            print(f"exit code {returncode}")
            print("terminating...")
            sys.exit(1)

        transfer.add_item(
            source_path=input_dir,
            destination_path=os.path.join(ARCHIVE_PATH, f"{experiment_name}.tar.gz"),
        )

    client.submit_transfer(transfer)
    print("Submitted transfer.")


def pull_experiments(client: globus_sdk.TransferClient, output_dir, *experiment_names):
    transfer = globus_sdk.TransferData(
        client,
        source_endpoint=FLUX_UUID,
        destination_endpoint=GREATLAKES_UUID,
        sync_level="checksum"
    )

    tars = []
    for experiment in experiment_names:
        experiment_tar = f"{experiment}.tar.gz"
        source_path = os.path.join(ARCHIVE_PATH, experiment_tar)
        tar_path = os.path.join(SCRATCH_PATH, experiment_tar)
        transfer.add_item(
            source_path=source_path,
            destination_path=tar_path
        )
        extraction_path = os.path.join(output_dir, experiment)
        tars.append(tar_path)

    resp = client.submit_transfer(transfer)
    task_id = resp.data["task_id"]
    print(f"submitted task {task_id}...")
    while not client.task_wait(task_id, timeout=10):
        print("waiting...")
    print("done.")

    task = client.get_task(task_id)
    status = task["status"]
    if status != "SUCCEEDED":
        print(f"Received an unexpected status for globus transfer: {status}")
        print("terminating...")
        sys.exit(1)

    def extract_experiment(tar_output):
        print(f"decompressing {tar_output} to {output_dir}...")
        returncode = subprocess.run(["tar", "-I", "pigz", "-xf", tar_output, "--directory", output_dir]).returncode
        return (tar_output, returncode)

    results_raw = Parallel(n_jobs=len(tars), prefer="threads")(map(delayed(extract_experiment), tars))

    # dumb typecasting stuff
    results: Iterable[Tuple[Tuple[str,str], bool]] = []
    if results_raw is not None:
        results = results_raw

    for tar_output, returncode in results:
        if not returncode == 0:
            print(f"Received a non-zero exit code while extracting {tar_output};")
            print(f"returncode {returncode}")
            print("terminating...")
            sys.exit(1)

    print("Pull complete.")


def clear_scratch():
    for path in glob(os.path.join(SCRATCH_PATH, "*")):
        try:
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        except Exception as e:
            print(e)
            sys.exit(1)


def verify_environment():
    if not os.path.isdir(SCRATCH_PATH):
        print(f"Turbo path does not exist: {SCRATCH_PATH}")
        return False
    if not os.path.isdir(TURBO_PATH):
        print(f"Turbo path does not exist: {TURBO_PATH}")
        return False

    return True


def verify_experiment_directory(path):
    if not os.path.isdir(path):
        print(f"Not a valid experiment directory: {path}")
        return False
    else:
        return True
