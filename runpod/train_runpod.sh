#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

FOLD="${1:?Indica un fold entre 0 y 4}"
case "$FOLD" in
    0|1|2|3|4) ;;
    *) echo "Fold invalido: $FOLD"; exit 1 ;;
esac

if [ "$FOLD" = "0" ]; then
    nnUNetv2_train 502 3d_fullres 0 \
        -p nnUNetResEncUNetMPlans --c
else
    nnUNetv2_train 502 3d_fullres "$FOLD" \
        -p nnUNetResEncUNetMPlans
fi
