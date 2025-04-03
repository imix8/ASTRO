# Use the official PyTorch GPU container with CUDA 12.6 support.
# Replace the tag with the correct one if needed. Ensure that the image supports ARM64.
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth2.0-py3

# Set the working directory
WORKDIR /app

# Install any additional system dependencies required by your application.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip if needed
RUN pip install --upgrade pip

# Copy the requirements file and install Python dependencies.
# Since the base image already includes PyTorch, you can omit it from requirements.txt if desired.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container.
COPY . /app

# Set the default command. Adjust the script as needed (e.g., image_predict.py, video_predict.py, etc.).
CMD ["python", "image_predict.py"]
CMD ["python", "webcam_predict.py"]
