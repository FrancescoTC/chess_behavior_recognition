#!/bin/bash
pip3 install git+https://github.com/huggingface/transformers accelerate

# It's highly recommanded to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]==0.0.8


python3 01_training_dataset.py
