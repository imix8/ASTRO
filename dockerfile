# Use an ARM64-compatible NVIDIA CUDA base image with CUDA 12.6 runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file and install Python dependencies.
# The --extra-index-url points to the PyTorch wheels built for CUDA 12.6.
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . /app

# Set the default command. Change the script name if needed.
CMD ["python3", "image_predict.py"]
