#!/bin/bash
#
# What do you get when you combine tar and glob? Gloop!
#
# This script was originally designed to migrate our data from local machines to
# Turbo.
#
# Given a glob, it will tar the inputs and output them to a provided destination.
#
#

PARAMS=""

while (( "$#" )); do
  case "$1" in
    -o|--out)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        OUT=$2
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
  dirname=$(dirname $path)
  basename=$(basename $path)
  output="$OUT$(basename $path).tar.gz"
  if [-f $output]; then
    continue
  fi
  while : ; do
    tar -I pigz -cvf $output -C $dirname $basename

    if [ $? -eq 0 ]; then
      break
    else
      rm -f $output
    fi
  done
done
