#!/bin/bash

docker run \
	-v /home/calzolari/chess:/workspace \
	--rm \
	--gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
	-it \
	calzolari_image \
	bash
