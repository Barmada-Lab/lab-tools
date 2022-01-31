#!/bin/bash
#
# Renames all files containing spaces with underscores
for file in *' '*; do
    mv -- "$file" "${file// /_}"
done
