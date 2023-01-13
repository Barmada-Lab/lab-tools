#! /bin/bash

set -e

export ORIGIN=$1
export DEST=$2
export SCRATCH=$3

tar_scp_rm() {
    dir=$1
    # clean ephemeral data
    rm -rf $dir/{analysis,processed_imgs}
    tar=$SCRATCH/$(basename $dir).tar.gz
    # archive the experiment
    tar -czf $tar $dir && scp $tar $DEST && rm -rf $tar $dir
}

export -f tar_scp_rm

find $1 -mindepth 1 -maxdepth 1 -type d | xargs -I {} bash -c 'tar_scp_rm "{}"'
