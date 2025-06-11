#!/bin/bash

# Usage: run_chuck3.sh input.in [output.out]

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [[ -z "$INPUT_FILE" ]]; then
  echo "Usage: $0 <input_file.in> [output_file.out]"
  exit 1
fi

# Resolve input file info
INPUT_FILE="$(realpath "$INPUT_FILE")"
INPUT_DIR="$(dirname "$INPUT_FILE")"
INPUT_BASENAME="$(basename "$INPUT_FILE")"

if [[ -n "$OUTPUT_FILE" ]]; then
  # Get absolute path only for the directory (realpath fails if file doesn't exist)
  OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
  mkdir -p "$OUTPUT_DIR"  # Create if needed
  OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
  OUTPUT_BASENAME="$(basename "$OUTPUT_FILE")"

  # Run docker with both mounts
  docker run --rm \
    --platform linux/amd64 \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    -w /input \
    chuck3-runner:latest \
    bash -c "chuck3 < $INPUT_BASENAME > /output/$OUTPUT_BASENAME"
else
  # Only input
  docker run --rm \
    --platform linux/amd64 \
    -v "$INPUT_DIR":/input \
    -w /input \
    chuck3-runner:latest \
    bash -c "chuck3 < $INPUT_BASENAME"
fi