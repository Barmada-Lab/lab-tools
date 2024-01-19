#!/bin/bash

set -e

help() {
    echo "Usage: $(basename $0) <experiment_name>"
}

if [[ -z $1 ]]; then
    help
    exit 1
fi

HOST=globus-xfer.arc-ts.umich.edu
ARCHIVE=/nfs/dataden/umms-sbarmada/experiments
DESTINATION=/nfs/turbo/umms-sbarmada/experiments/
TEMP=/scratch/sbarmada_root/sbarmada0/$USER/dataden

mkdir -p $TEMP

EXP_NAME=$1
FILENAME="$EXP_NAME.tar.gz"
echo "get '$ARCHIVE/$FILENAME' '$TEMP'" | sftp $USER@$HOST
tar -xvf "$TEMP/$FILENAME" -C $DESTINATION
touch "$DESTINATION/$EXP_NAME"
rm "$TEMP/$FILENAME"
