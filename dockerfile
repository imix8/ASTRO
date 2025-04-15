FROM nvcr.io/nvidia/l4t-base:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    # PyTorch dependencies
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    ninja-build \
    # Update CA certificates store
    && update-ca-certificates --fresh \
    # Attempt to remove system python3-opencv if it exists
    && apt-get purge -y --auto-remove python3-opencv || true \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build and install CMake 3.18.4
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/archive/refs/tags/v3.18.4.tar.gz && \
    tar -xzvf v3.18.4.tar.gz && \
    cd CMake-3.18.4 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/*

# Build and install Python 3.9.18
ARG PYTHON_VERSION=3.9.18 # Use a specific recent 3.9.x version
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make altinstall && \
    # Verify installation
    /usr/local/bin/python3.9 --version && \
    /usr/local/bin/pip3.9 --version && \
    # Cleanup
    cd / && \
    rm -rf /tmp/*

# Set build arguments for PyTorch and Jetson Nano support
ARG PYTORCH_VERSION=v1.13.0 # Using older version likely more compatible with JP4.4 build tools
ARG TORCH_CUDA_ARCH_LIST="5.3" # Compute Capability for Jetson Nano

# Upgrade pip and prepare for PyTorch build
RUN /usr/local/bin/pip3.9 install --no-cache-dir --upgrade pip setuptools wheel

# Clone PyTorch repository
WORKDIR /opt
RUN git clone --recursive --branch ${PYTORCH_VERSION} https://github.com/pytorch/pytorch.git

# Install PyTorch requirements and fix potential dependency issues
RUN cd pytorch && \
    /usr/local/bin/pip3.9 install --no-cache-dir -r requirements.txt && \
    # Pin NumPy to a version compatible with PyTorch 1.10.x to avoid C API issues
    /usr/local/bin/pip3.9 install --no-cache-dir "numpy<1.22" && \
    # Fix for setup.py issue with setuptools >= 59.6.0 (common on newer pip)
    /usr/local/bin/pip3.9 install --no-cache-dir "setuptools<59.6.0"

# Build and install PyTorch with the necessary environment variables
RUN cd pytorch && \
    export PYTORCH_BUILD_VERSION=$(echo "${PYTORCH_VERSION}" | sed 's/^v//' | cut -d. -f1,2) && \
    export PYTORCH_BUILD_NUMBER=1 && \
    export USE_NCCL=0 && \
    export USE_QNNPACK=1 && \
    export USE_PYTORCH_QNNPACK=1 && \
    export USE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \
    export BUILD_TEST=0 && \
    export VERBOSE=1 && \
    export MAX_JOBS=8 && \
    /usr/local/bin/python3.9 setup.py install

# Clone and build torchvision 0.14.0 from source
WORKDIR /opt

# Install newer CMake version to support torchvision
ARG CMAKE_MIN_VERSION=3.22 # Minimum version needed
RUN /usr/local/bin/pip3.9 install --no-cache-dir "cmake>=${CMAKE_MIN_VERSION}"

ARG TORCHVISION_VERSION=v0.14.0
WORKDIR /opt
RUN git clone --recursive --branch ${TORCHVISION_VERSION} --depth 1 https://github.com/pytorch/vision.git vision
RUN cd vision && \
    export BUILD_VERSION=$(echo "${TORCHVISION_VERSION}" | sed 's/^v//') && \
    # export MAX_JOBS=$(nproc) && \
    export MAX_JOBS=8 && \
    /usr/local/bin/python3.9 setup.py install && \
    cd / && rm -rf /opt/vision

# Cleanup torch build files
RUN rm -rf /opt/pytorch

# Install rfdetr dependencies separately to avoid onnx
RUN /usr/local/bin/pip3.9 install --no-cache-dir \
    supervision \
    matplotlib \
    cython \
    pycocotools \
    "torch==1.13.0" \
    "torchvision==0.14.0" \
    fairscale \
    scipy \
    timm \
    tqdm \
    "numpy<1.22" \
    accelerate \
    transformers \
    peft \
    ninja \
    einops \
    pandas \
    pylabel \
    polygraphy \
    open_clip_torch \
    rf100vl \
    pydantic && \
    rm -rf /root/.cache/pip

# Install rfdetr without dependencies
RUN export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" && \
    /usr/local/bin/pip3.9 install --no-cache-dir --no-deps \
    rfdetr && \
    rm -rf /root/.cache/pip

# Install Jetson.GPIO
RUN git clone https://github.com/NVIDIA/jetson-gpio.git /opt/Jetson.GPIO && \
    cd /opt/Jetson.GPIO && \
    /usr/local/bin/python3.9 setup.py install && \
    cd / && rm -rf /opt/Jetson.GPIO

WORKDIR /workspace

# Copy necessary files
COPY image_predict.py /workspace/
COPY webcam_predict_jetson.py /workspace/
COPY webcam_predict_laptop.py /workspace/
COPY logs/ /workspace/logs/
COPY dataset/ /workspace/dataset/

CMD ["bash"]