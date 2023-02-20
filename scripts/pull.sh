#!/bin/bash

set -e

help() {
    echo "Usage: $(basename $0) <experiment_name>"
}

if [ -z $1 ]; then
    help
    exit 1
fi

HOST=globus-xfer.arc-ts.umich.edu
ARCHIVE=/nfs/dataden/umms-sbarmada/experiments
DESTINATION=/scratch
TEMP=/scratch/sbarmada_root/sbarmada0/$USER/dataden

mkdir -p $TEMP

filename="$1.tar.gz"
scp $USER@$HOST:$ARCHIVE/$filename $TEMP
tar -xvf $TEMP/$filename -C $DESTINATION
rm $TEMP/$filename
