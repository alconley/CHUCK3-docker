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

# Make sure chuck3 is executable
RUN chmod +x chuck3

# Default command (does nothing on its own, overridden by script)
CMD ["/bin/bash"]