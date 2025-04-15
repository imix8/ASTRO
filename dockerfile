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
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libeigen3-dev \
    libopenmpi-dev \
    libomp-dev \
    # Update CA certificates store
    && update-ca-certificates --fresh \
    # Attempt to remove system python3-opencv if it exists
    && apt-get purge -y --auto-remove python3-opencv || true \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build and install Python 3.9.18 (keeping this custom Python build)
ARG PYTHON_VERSION=3.9.18
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


# Upgrade pip and install Python build requirements from instructions
RUN /usr/local/bin/pip3.9 install --no-cache-dir --upgrade pip wheel && \
    /usr/local/bin/pip3.9 install --no-cache-dir \
    mock \
    pillow \
    testresources \
    setuptools==58.3.0 \
    scikit-build

# Install newer CMake via pip needed for Torchvision build
ARG CMAKE_MIN_VERSION=3.22
RUN /usr/local/bin/pip3.9 install --no-cache-dir "cmake>=${CMAKE_MIN_VERSION}"

# Optional cleanup of Python build dependencies to reduce image size
RUN apt-get purge -y \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev && \
    apt-get autoremove -y && apt-get clean

# Set build arguments for PyTorch and Jetson Nano support
ARG PYTORCH_VERSION=v1.13.0
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"

# Clone PyTorch repository
WORKDIR /opt
RUN git clone --recursive --branch ${PYTORCH_VERSION} --depth=1 https://github.com/pytorch/pytorch.git

# Build and install PyTorch using the specific instruction flags
RUN cd pytorch && \
    export PYTORCH_BUILD_VERSION=$(echo "${PYTORCH_VERSION}" | sed 's/^v//') && \
    export PYTORCH_BUILD_NUMBER=1 && \
    export BUILD_CAFFE2_OPS=OFF && \
    export USE_FBGEMM=OFF && \
    export USE_FAKELOWP=OFF && \
    export BUILD_TEST=OFF && \
    export USE_MKLDNN=OFF && \
    export USE_NNPACK=OFF && \
    export USE_XNNPACK=OFF && \
    export USE_QNNPACK=OFF && \
    export USE_PYTORCH_QNNPACK=OFF && \
    export USE_CUDA=ON && \
    export USE_CUDNN=ON && \
    export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2" && \
    export USE_NCCL=OFF && \
    export USE_SYSTEM_NCCL=OFF && \
    export USE_OPENCV=OFF && \
    export MAX_JOBS=4 && \
    export CUDACXX=/usr/local/cuda/bin/nvcc && \
    /usr/local/bin/python3.9 setup.py bdist_wheel

# Install the built PyTorch wheel
RUN cd pytorch/dist && \
    /usr/local/bin/pip3.9 install --no-cache-dir torch-*.whl && \
    cd / && rm -rf /opt/pytorch


# Clone and build TorchVision
ARG TORCHVISION_VERSION=v0.14.0
WORKDIR /opt
RUN git clone --recursive --branch ${TORCHVISION_VERSION} --depth 1 https://github.com/pytorch/vision.git vision
RUN cd vision && \
    export BUILD_VERSION=$(echo "${TORCHVISION_VERSION}" | sed 's/^v//') && \
    export MAX_JOBS=8 && \
    /usr/local/bin/python3.9 setup.py install && \
    cd / && rm -rf /opt/vision

# Install rfdetr dependencies
# Removed torch and torchvision from this list as they are already installed from source
RUN /usr/local/bin/pip3.9 install --no-cache-dir \
    supervision \
    matplotlib \
    cython \
    pycocotools \
    # torch==1.13.0 # Already installed from wheel
    # torchvision==0.14.0 # Already installed from source
    fairscale \
    scipy \
    timm \
    tqdm \
    # numpy<1.22 # Already installed
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

# Install Jetson.GPIO using Python 3.9
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