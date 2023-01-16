from pathlib import Path
import os

SCRATCH_PATH = Path("/scratch/sbarmada_root/sbarmada0") / os.environ.get("USER") / ".lab_tools"
EXPERIMENT_DIRECTORY = Path("/nfs/turbo/umms-sbarmada/experiments")
ILASTIK_BINARY = Path("/nfs/turbo/umms-sbarmada/shared/ilastik-1.4.0b27-Linux/run_ilastik.sh")
ILASTIK_CLASSIFIERS = Path("/nfs/turbo/umms-sbarmada/shared/classifiers")
MODEL_DIR = Path("/nfs/turbo/umms-sbarmada/shared/models")
CPUS = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
RSCRIPT_DIR = Path(os.environ.get("HOME")) / "Repos" / "lab-tools" / "rscripts"