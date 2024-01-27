#!/bin/bash

python_interpreter="/home/maxence/miniconda3/envs/tf/bin/python"

# Define the list of arguments
arguments=(
    "metric_set2_models_bgr_v3.json"
)

# Loop through the arguments and run the Python script for each file
for file in "${arguments[@]}"; do
    $python_interpreter MEGA_TEST.py -p "$file"
done
