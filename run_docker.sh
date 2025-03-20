#!/bin/bash

PHYS_DIR="/home/calzolari/chess"

docker run \
	-v "$PHYS_DIR":/workspace \
	--rm \
	--gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
	calzolari_image \
	"/workspace/train.sh"
