from importlib import metadata
import subprocess
import sys

from github import Github

from cytomancer.utils import get_user_confirmation


DIST_NAME = "cytomancer"
REPO_NAME = f"Barmada-Lab/{DIST_NAME}"

gh = Github()


def get_local_version():
    return metadata.version(DIST_NAME)


def get_latest_version():
    repo = gh.get_repo(REPO_NAME)
    release = repo.get_latest_release()
    return release.tag_name


def get_latest_whl_asset():
    repo = gh.get_repo(REPO_NAME)
    release = repo.get_latest_release()

    whls = []
    for asset in release.get_assets():
        if asset.name.startswith(DIST_NAME) and asset.name.endswith(".whl"):
            whls.append(asset)

    if not whls:
        raise ValueError("No .whl assets found in latest release")
    elif len(whls) > 1:
        raise ValueError("Multiple .whl assets found in latest release")

    whl = whls[0]
    return whl


def pipx_upgrade_whl(whl_url: str):
    # when using pipx, we need to uninstall the old version before installing the new one
    print()
    print(f"uninstalling {DIST_NAME}...")
    subprocess.run(['pipx', 'uninstall', DIST_NAME])
    print()
    subprocess.check_call(['pipx', 'install', whl_url])


def check_for_updates():
    local_version = get_local_version()
    latest_version = get_latest_version()

    if local_version == latest_version:
        return

    print()
    print(f"A new release of {DIST_NAME} is available ({local_version} -> {latest_version})")
    release_url = gh.get_repo(REPO_NAME).get_latest_release().html_url
    print(f"read the release notes at: {release_url}")
    print()

    if get_user_confirmation("Would you like to upgrade?", default="y"):
        whl = get_latest_whl_asset()
        pipx_upgrade_whl(whl.browser_download_url)
        sys.exit(0)
