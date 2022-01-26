import globus_sdk
from globus_sdk.authorizers import access_token
from . import helpers
from . import core

import dotenv
import argparse
import sys
import os

root_parser = argparse.ArgumentParser(description="ARchive Tool")
subparsers = root_parser.add_subparsers(required=True, dest="command")

auth_parser = subparsers.add_parser("auth", help="Authorize an application")
auth_parser.add_argument("client_id", type=str)

list_parser = subparsers.add_parser("list", help="List experiments available for retrieval")

pull_parser = subparsers.add_parser("pull", help="Retrieve an experiment")
pull_parser.add_argument("experiment", help="name of the experiment to fetch", type=str)
pull_parser.add_argument("output_dir", help="output directory for download", type=os.path.abspath)

push_parser = subparsers.add_parser("push", help="Archive an experiment")
push_parser.add_argument("experiments", nargs="+", help="path to experiment", type=os.path.abspath)


def main():

    args = root_parser.parse_args()
    command = args.command
    secret_path = os.path.join(os.path.expanduser("~"), ".globusconf")

    if command == "auth":

        client_id = args.client_id
        auth_response = helpers.authorize(client_id)
        print(auth_response)

        auth_refresh_token = auth_response.by_resource_server[core.AUTH_SERVER][core.REFRESH_TOKEN]
        transfer_refresh_token = auth_response.by_resource_server[core.TRANSFER_SERVER][core.REFRESH_TOKEN]
        #connect_server_refresh_token = auth_response.by_resource_server[core.TRANSFER_SERVER][core.REFRESH_TOKEN]
        secrets = {
            "CLIENT_ID": client_id,
            "AUTH_REFRESH_TOKEN": auth_refresh_token,
            "TRANSFER_REFRESH_TOKEN": transfer_refresh_token
        }

        with open(secret_path, "wt") as f:
            f.writelines([f"{key}={value}\n" for key, value in secrets.items()])
            print(f"wrote client secrets to {secret_path}")

    else:

        if not os.path.exists(secret_path):
            print(f"Secrets file not found at {secret_path}; have you run the 'auth' command yet?")
            sys.exit(1)

        config = dotenv.dotenv_values(secret_path)
        client_id = config["CLIENT_ID"]
        refresh_token = config["TRANSFER_REFRESH_TOKEN"]
        client = helpers.transfer_client(client_id, refresh_token)

        if command == "push":
            core.push_experiments(client, *args.experiments)
        elif command == "list":
            core.list_experiments(client)
        elif command == "pull":
            core.pull_experiments(client, args.output_dir, args.experiment)


if __name__ == "__main__":
    main()
