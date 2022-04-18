#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR && poetry run ipython kernel install --user --name=barma

JUPYTER_CONF=$HOME/.jupyter/jupyter_lab_config.py
if [[ ! -f $JUPYTER_CONF ]]; then
    NB_DIR=$SCRIPT_DIR/notebooks
    echo "c.ServerApp.root_dir = '$NB_DIR'" > $JUPYTER_CONF
fi;
