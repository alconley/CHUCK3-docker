#!/bin/bash

# Usage: ./run_chuck3.sh input.in [output.out]

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [[ -z "$INPUT_FILE" ]]; then
  echo "Usage: $0 <input_file.in> [output_file.out]"
  exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
  # No output file — pipe result to terminal
  docker run --rm \
    --platform linux/amd64 \
    -v "$PWD":/app \
    -w /app \
    chuck3-runner \
    bash -c "./chuck3 < $INPUT_FILE"
else
  # Output file specified — write to it and also show it
  docker run --rm \
    --platform linux/amd64 \
    -v "$PWD":/app \
    -w /app \
    chuck3-runner \
    bash -c "./chuck3 < $INPUT_FILE > $OUTPUT_FILE && cat $OUTPUT_FILE"
fi