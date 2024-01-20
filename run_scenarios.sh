#!/bin/bash

python_interpreter="/home/maxence/miniconda3/envs/tf/bin/python"

# Define the list of arguments
arguments=(
    "metric_set1_models_bgr.json"
    "metric_set2_algo.json"
    "metric_set2_models_bgr.json"
    "metric_set2_models_bgrds.json"
)

# Loop through the arguments and run the Python script for each file
for file in "${arguments[@]}"; do
    $python_interpreter MEGA_TEST.py -p "$file"
done
