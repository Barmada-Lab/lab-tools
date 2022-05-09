#!/bin/bash
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

compress() {
    if [[ $(identify -verbose $1 | grep Compression | awk '{ print $2 }') == 'None' ]]; then
        echo "compressing $1"
        convert $1 -compress lzw $1
    fi;
}

# gotta export and call from a subshell because xargs won't pick up defined
# funtions for some reason
export -f compress

for path in $PARAMS
do
    find $1 -name *.tif | xargs -l -P $CPUS bash -c 'compress $@' _
done
