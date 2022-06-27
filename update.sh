#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
SCRIPT_FILE=`basename "$0"`
SCRIPT_PATH=$SCRIPT_DIR/$SCRIPT_FILE

setup() {
    cd $SCRIPT_DIR/notebooks
    poetry update
    poetry run ./nbkgen.sh
}

cd $SCRIPT_DIR

ORIGINAL_MD5=$(md5sum $SCRIPT_PATH | cut -d' ' -f1)
git reset --hard origin/main
git pull
NEW_MD5=$(md5sum $SCRIPT_PATH | cut -d' ' -f1)

if [[ $ORIGINAL_MD5 == $NEW_MD5 ]]; then
    setup
else
    echo ""
    echo "Update script modified when pulling code. Running new update script..."
    /bin/bash $SCRIPT_FILE
fi
