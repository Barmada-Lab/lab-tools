from pathlib import Path
import os

SCRATCH_PATH = Path("/scratch/sbarmada_root/sbarmada0") / os.environ.get("USER") / ".lab_tools"
if not SCRATCH_PATH.exists():
    SCRATCH_PATH.mkdir()
EXPERIMENT_DIRECTORY = Path("/nfs/turbo/umms-sbarmada/experiments")
ILASTIK_BINARY = Path("/nfs/turbo/umms-sbarmada/shared/ilastik-1.4.0b27-Linux/run_ilastik.sh")
ILASTIK_CLASSIFIERS = Path("/nfs/turbo/umms-sbarmada/shared/classifiers")
CPUS = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))