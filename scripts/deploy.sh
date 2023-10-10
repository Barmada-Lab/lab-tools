#!/bin/bash

set -e

TARGET_DIR=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
LABTOOLS_DIR=$SCRIPT_DIR/..
rsync -rv $LABTOOLS_DIR $TARGET_DIR --exclude=".*" --exclude=__pycache__
