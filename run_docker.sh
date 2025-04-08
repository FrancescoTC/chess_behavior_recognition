#!/bin/bash

PHYS_DIR="/home/calzolari/chess"
HF_TOKEN=$(./get_token.sh)

docker run \
	-v "$PHYS_DIR":/workspace \
	--rm \
	--gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
	-e HF_TOKEN="$HF_TOKEN" \
	calzolari_image2 \
	"/workspace/train.sh"
