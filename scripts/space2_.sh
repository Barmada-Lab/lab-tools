#!/bin/bash
#
# Renames all files containing spaces with underscores
for dir in $@; do
    for file in $dir*; do
        if [[ $file =~ .*' '.* ]]; then
            mv -- "$file" "${file// /_}"
        fi
    done
done
