import subprocess
import shutil
import time
import sys
import os

from glob import glob

from . import core

def print_usage(program_name):
    print(f"Usage: masa COMMAND OPTIONS")
    print("commands: {up, down, deploy}")

def print_deploy_usage(program_name):
    print(f"Usage: masa SOURCE_PATH MASA_IDX")
    print(f"Example: {program_name} g:/mbiera0024 11")

def parse_args(args: list):
    try:
        command = args.pop(1)
    except IndexError:
        print_usage(args[0])
        sys.exit(1)
    if command == "up":
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
        start_server("/home/www-data/masa-cc")
    elif command == "down":
        subprocess.run(["pkill", "-f", "server.py"])
    elif command == "deploy":
        try:
            masa_idx = int(args[2])
            path = os.path.relpath(args[1])
            if not os.path.isdir(path):
                raise Exception(f"{path} is not an experiment directory")
            arg_dict = {
                "command": command,
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
        except IndexError:
            print_deploy_usage(args[0])
            raise Exception("Argument parsing failed")
        except Exception as e:
            print(e)
            raise Exception("Unhandled exception")
    else:
        print_usage(args[0])


def check_targets(data_loc, masa_name, exp_name):
    masa_path = os.path.join(data_loc, masa_name)
    data_path = os.path.join(masa_path, "data", "stacks")
    roi_glob = glob(os.path.join(masa_path, "server", "*-rois"))

    if os.path.exists(masa_path):
        print(f"{masa_path} already exists; running this command will overwrite all data associated with that experiment.")
        print("Please ensure rois have been saved elsewhere; the following paths will be deleted:")
        print("------------------")
        print(data_path)
        for match in roi_glob:
            print(match)

        choice = input("Confirm? [y/N] ").lower()
        if choice in ["y", "ye", "yes"]:
            if os.path.isdir(data_path):
                shutil.rmtree(data_path)

            for match in roi_glob:
                shutil.rmtree(match)

        elif choice in ["n", ""]:
            print("exiting...")
            sys.exit(0)
        else:
            print("I'm not sure what you meant by that...")
            print("it was a simple question...")
            sys.exit(1)


def main():
    try:
        args = parse_args(sys.argv)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
