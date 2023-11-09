#!/bin/bash

# Check if the folder path is provided as a parameter
if [ $# -ne 1 ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Get the folder path from the first parameter
folder_path="$1"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
  echo "Error: Folder '$folder_path' does not exist."
  exit 1
fi

# Loop through each file in the folder
for file in "$folder_path"/*; do
  if [ -f "$file" ]; then
    echo "Processing file: $file"
    python3 image_reduction.py "$file"
  fi
done
