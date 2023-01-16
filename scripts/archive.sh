#! /bin/bash

set -e

export ORIGIN=$1
export DEST=$2
export SCRATCH=$3
export LOCAL=$4

tar_scp_rm() {
    set -e

    dir=$1

    # clean ephemeral data
    rm -rf $dir/{analysis,processed_imgs}

    dirname=$(basename $dir)
    tar=$SCRATCH/${dirname}.tar.gz
    cached=$LOCAL/$dirname

    # Archive the experiment
    tar -czf $tar $dir && scp $tar $DEST && rm -rf $tar $dir $cached

}

export -f tar_scp_rm

find $1 -mindepth 1 -maxdepth 1 -type d -mtime +60 -exec bash -c 'tar_scp_rm "$0"' {} \;
