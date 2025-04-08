#!/bin/bash

HF_TOKEN=$(./get_token.sh)

docker run \
	-v /home/calzolari/chess:/workspace \
	--rm \
	--gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
	-it \
	-e HF_TOKEN="$HF_TOKEN" \
	calzolari_image2 \
	bash
