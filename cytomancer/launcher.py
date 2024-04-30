from pathlib import Path
from importlib import metadata
import subprocess
import tempfile
import sys

from github import Github
import requests


DISTRIBUTION_NAME = "cytomancer"

REPO_NAME = "Barmada-Lab/cytomancer-tools"

gh = Github()


def get_local_version():
    return metadata.version(DISTRIBUTION_NAME)


def get_latest_version():
    repo = gh.get_repo(REPO_NAME)
    release = repo.get_latest_release()
    return release.tag_name


def update_available():
    return get_local_version() != get_latest_version()


def download_latest_whl(download_dir: Path = Path.cwd()):
    repo = gh.get_repo(REPO_NAME)
    release = repo.get_latest_release()

    whls = []
    for asset in release.get_assets():
        if asset.name.startswith(DISTRIBUTION_NAME) and asset.name.endswith(".whl"):
            whls.append(asset)

    if not whls:
        raise ValueError("No .whl assets found in latest release")
    elif len(whls) > 1:
        raise ValueError("Multiple .whl assets found in latest release")

    whl = whls[0]
    response = requests.get(whl.browser_download_url)

    path = download_dir / whl.name
    with open(path, "wb") as f:
        f.write(response.content)

    return path


def pipx_install_whl(whl_file: Path):
    subprocess.check_call([sys.executable, '-m', 'pipx', 'install', whl_file])


def install_latest_whl():
    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        whl_file = download_latest_whl(download_dir)
        pipx_install_whl(whl_file)
