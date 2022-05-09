#!/bin/bash
#
# Cleanup script automagically
PARAMS=""

CPUS=1

while (( "$#" )); do
  case "$1" in
    -c|--cpus)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        CPUS=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done # set positional arguments in their proper place

eval set -- "$PARAMS"

prep() {
    find $1 -name *.tif -exec convert '{}' -compress lzw '{}' \;
    find $1 -maxdepth 2 -name processed_imgs -type d -mtime +30 -prune -exec rm -rf '{}' \;
    echo "Finished prepping $1"
}

# gotta export and call from a subshell because xargs won't pick up defined
# funtions for some reason
export -f prep

for path in $PARAMS
do
    ls $path | xargs -l -P $CPUS bash -c 'prep $@' _
    find $path -maxdepth 0 -type d -mtime +60 | xargs art push
done
