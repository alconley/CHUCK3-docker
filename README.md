# CHUCK3-docker

This project wraps the [CHUCK-3](https://inis.iaea.org/records/29j6s-xfg63), a nuclear reaction code for nuclear scattering amplitude and collision cross-sections by couple channel, in a Docker container so it can run easily on macOS, Windows, or Linux — even when the binary is built for Ubuntu x86_64.

---

## 📁 Contents

```
chuck3_runner/
├── Dockerfile           # Docker environment definition
├── chuck3               # Precompiled binary (Linux x86_64)
├── run_chuck3.sh        # Script to run simulations
├── example_input.in     # Optional sample input file
```

---

## ✅ Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop)

---

## 🛠️ Setup Instructions

### Step 1: Install Docker

Download and install Docker Desktop:
> https://www.docker.com/products/docker-desktop

---

### Step 2: Build the Docker Image

Open a terminal in this directory and run:

```bash
docker build --platform linux/amd64 -t chuck3-runner .
```

This builds a container with Ubuntu 18.04 and required libraries for `chuck3`.

---

### Step 3: Run the Program

Use the provided script:

```bash
./run_chuck3.sh path/to/input.in                 # Runs chuck3 and prints output to terminal
./run_chuck3.sh path/to/input.in path/to/out.out # Runs chuck3 and saves output to file
```

---

## 🐚 `run_chuck3.sh`

```bash
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
```

---

## 🧱 `Dockerfile`

```Dockerfile
# Use Ubuntu 18.04 because it includes libgfortran.so.3
FROM ubuntu:18.04

# Set working directory inside the container
WORKDIR /app

# Install required packages including gfortran runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libgfortran3 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from host into /app
COPY . .

# Make sure chuck3 is executable and move it to PATH
RUN chmod +x chuck3 && mv chuck3 /usr/local/bin/

# Default command (does nothing on its own, overridden by script)
CMD ["/bin/bash"]
```

---

## 🧽 Cleanup

To remove the image:

```bash
docker rmi chuck3-runner
```
