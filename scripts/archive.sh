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

    base=$dir/..
    dirname=$(basename $dir)
    tar=$SCRATCH/${dirname}.tar.gz
    cached=$LOCAL/$dirname

    # Archive the experiment
    tar -czf $tar -C $base $dirname && scp $tar $DEST && rm -rf $tar $dir $cached

}

export -f tar_scp_rm

find $ORIGIN -mindepth 1 -maxdepth 1 -type d -mtime +60 | head -30 | xargs -n 1 -I {} bash -c 'tar_scp_rm "$0"' {}
