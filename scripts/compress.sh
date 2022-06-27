#!/bin/bash
PARAMS=""

stop_script=false
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
    stop_script=false
    trap "stop_script=true" SIGINT
    echo "compressing $1"

    convert $1 -compress lzw $1._tmp
    if (( $? != 0 )); then
      if [[ -f $1._tmp ]]; then
          rm $1._tmp
      fi;
      echo "encountered an issue while compressing"
      return 0
    fi;
    if [ $stop_script = true ]; then
        exit 0
    fi;
    initial_size=$(stat --printf="%s" $1)
    compressed_size=$(stat --printf="%s" $1._tmp)

    # Sometimes lzw will make files bigger :^)
    if (( compressed_size < initial_size )); then
      mv $1._tmp $1
    else
      rm $1._tmp
    fi;
}

# gotta export and call from a subshell because xargs won't pick up defined
# funtions for some reason
export -f compress

echo "running at x$CPUS parallelism"
for path in $PARAMS
do
    # Skip over white_light subdir because it's always gonna produce bigger images when compressed with LZW
    find $1 -name *.tif -not -path "*/white_light/*" | xargs -l -P $CPUS bash -c 'compress $@' _
done
