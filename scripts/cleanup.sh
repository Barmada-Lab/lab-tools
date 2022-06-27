#!/bin/bash
#
# Automagically clean and archive things
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

for path in $PARAMS
do
    find $path -mindepth 2 -maxdepth 2 -name processed_imgs -type d -mtime +30 | xargs -l -P $CPUS rm -rf '{}' \;
    find $path -mindepth 1 -maxdepth 1 -type d -mtime +60 | xargs art push
done
