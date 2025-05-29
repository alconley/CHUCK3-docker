#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# start timer
SECONDS=0

usage() {
  cat <<EOF
Usage:
  $0 [--parallel N] <infile1.in|pattern1> [infile2.in|pattern2 …]

Options:
  --parallel N   run up to N jobs in parallel (requires GNU parallel).
EOF
  exit 1
}

PARALLEL=0
JOBS=1

# parse --parallel
if [[ "${1-}" == "--parallel" ]]; then
  [[ -z "${2-}" ]] && usage
  PARALLEL=1
  JOBS="$2"
  shift 2
fi

[[ $# -lt 1 ]] && usage

# expand globs
infiles=()
for pat in "$@"; do
  for f in $pat; do
    infiles+=( "$f" )
  done
done
(( ${#infiles[@]} == 0 )) && { echo "Error: no input files found" >&2; exit 1; }

run_single() {
  local infile="$1"
  local outfile="${infile%.*}.out"
  docker run --rm \
    --platform linux/amd64 \
    -v "$PWD":/app \
    -w /app \
    chuck3-runner \
    bash -c "./chuck3 < '$infile' > '$outfile'"
}

# Fragment a single file if in parallel mode and only one input
if (( PARALLEL )) && (( ${#infiles[@]} == 1 )); then
  original="${infiles[0]}"
  base_dir=$(dirname "$original")
  base_name=$(basename "$original" .in)

  echo "Fragmenting $original for parallel execution..."

  # Strip final '9' line (if it exists)
  sed '${
    /^9$/d
  }' "$original" > "${original}.tmp"

  # Extract all cards into temporary file, separating by +00+00
  tmp_cards=()
  current_card=""
  while IFS= read -r line; do
    current_card+="$line"$'\n'
    if [[ "$line" == *"+00+00" ]]; then
      tmp_cards+=("$current_card")
      current_card=""
    fi
  done < "${original}.tmp"

  total_cards=${#tmp_cards[@]}
  cards_per_chunk=$(( (total_cards + JOBS - 1) / JOBS ))  # ceil division

  echo "Total cards: $total_cards → $JOBS chunks, ~$cards_per_chunk per chunk"

  # Write cards into chunk files
  chunk_files=()
  for ((i=0; i<JOBS; i++)); do
    chunk_file="${base_dir}/${base_name}_chunk_${i}.in"
    chunk_files+=("$chunk_file")
    : > "$chunk_file"  # truncate
  done

  for ((i=0; i<total_cards; i++)); do
    chunk_index=$((i / cards_per_chunk))
    [[ $chunk_index -ge $JOBS ]] && chunk_index=$((JOBS - 1))  # catch any overflow
    echo -n "${tmp_cards[i]}" >> "${chunk_files[$chunk_index]}"
  done

  # Add terminator to each file
  for file in "${chunk_files[@]}"; do
    echo "9" >> "$file"
  done

  rm -f "${original}.tmp"

  infiles=( "${chunk_files[@]}" )

  echo "Created ${#infiles[@]} fragments."
fi

if (( PARALLEL )); then
  if ! command -v parallel &>/dev/null; then
    echo "Error: GNU parallel not found. Install it or remove --parallel." >&2
    exit 1
  fi

  export -f run_single
  parallel -j "$JOBS" --bar run_single ::: "${infiles[@]}"

  # Combine into the original name
  original_base="${infiles[0]%%_chunk_*}.in"
  final_out="${original_base%.in}.out"

  echo "Combining into $final_out"
  : > "$final_out"
  for infile in "${infiles[@]}"; do
    cat "${infile%.*}.out" >> "$final_out"
  done

  echo "Cleaning up fragment input/output files"
  for infile in "${infiles[@]}"; do
    rm -f "$infile" "${infile%.*}.out"
  done

  echo "Done. See $final_out"

else
  for infile in "${infiles[@]}"; do
    echo "Processing $infile → ${infile%.*}.out"
    run_single "$infile"
  done
fi

echo "Total elapsed time: ${SECONDS}s"