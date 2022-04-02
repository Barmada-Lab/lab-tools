import subprocess
import argparse
import pathlib
import time
import shutil
import sys
import os

from glob import glob

from . import core

def check_targets(data_loc, masa_name, exp_name):
    masa_path = os.path.join(data_loc, masa_name)
    data_path = os.path.join(masa_path, "data", "stacks")
    roi_glob = glob(os.path.join(masa_path, "server", "*-rois"))

    if os.path.exists(masa_path):
        print(f"{masa_path} already exists; running this command will overwrite all data associated with that experiment.")
        print("Please ensure rois have been saved elsewhere; the following paths will be deleted:")
        print("------------------")
        print(data_path)
        for match_ in roi_glob:
            print(match_)

        choice = input("Confirm? [y/N] ").lower()
        if choice in ["y", "ye", "yes"]:
            if os.path.isdir(data_path):
                shutil.rmtree(data_path)

            for match_ in roi_glob:
                shutil.rmtree(match_)

        elif choice in ["n", ""]:
            print("exiting...")
            sys.exit(0)
        else:
            print("I'm not sure what you meant by that...")
            print("it was a simple question...")
            sys.exit(1)

def up_handler(args):
    def start_server(directory):
        with open(os.path.join(directory, "access.log"), "wb") as stdout, \
             open(os.path.join(directory, "error.log"), "wb") as stderr:
            subprocess.Popen(['python', "server.py"],
                             cwd=directory,
                             env={"PATH":os.getenv("PATH", "")},
                             stdout=stdout,
                             stderr=stderr)
    for directory in glob("/data/masa/masa*/server"):
        start_server(directory)
    time.sleep(5) # waiting for servers to actually be up and discoverable to masa-cc
    start_server("/home/www-data/masa-cc")

def down_handler(args):
    subprocess.run(["pkill", "-f", "server.py"])

def deploy_handler(args):
    try:
        masa_idx = args.masa_idx
        path = args.path
        if not os.path.isdir(path):
            raise Exception(f"{path} is not an experiment directory")
        arg_dict = {
            "scramble": False,
            "path": path,
            "masa_idx": masa_idx,
            "masa_name": f"masa{masa_idx}",
            "exp_name": os.path.basename(path),
            "data_loc": os.path.join(os.path.sep, "data", "masa"),
            "server_loc" : os.path.join(os.path.sep, "home", "rmiguez")
        }
        check_targets(arg_dict["data_loc"], arg_dict["masa_name"], arg_dict["exp_name"])
        core.deploy(arg_dict)
    except Exception as e:
        print(e)
        raise Exception("Unhandled exception")

def main():
    root = argparse.ArgumentParser("masa", description="masa management service")
    subparsers = root.add_subparsers(dest='cmd')
    subparsers.add_parser("up")
    subparsers.add_parser("down")
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("path", type=os.path.relpath)
    deploy_parser.add_argument("masa idx", type=int)

    args = root.parse_args()
    match args.cmd:
        case "up": up_handler(args)
        case "down": down_handler(args)
        case "deploy": deploy_handler(args)

if __name__ == "__main__":
    main()
