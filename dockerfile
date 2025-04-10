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

# RUN /usr/local/bin/pip3.9 install --no-cache-dir \
#     matplotlib \
#     supervision \
#     rfdetr && \
#     rm -rf /root/.cache/pip

# WORKDIR /workspace

# COPY image_predict.py /workspace/
# COPY logs/ /workspace/logs/
# COPY dataset/ /workspace/dataset/

# # Verify PyTorch install and CUDA availability from Python 3.9
# RUN /usr/local/bin/python3.9 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); \
#     has_cuda = torch.cuda.is_available(); \
#     print(f'CUDA Version: {torch.version.cuda if has_cuda else "N/A"}'); \
#     print(f'cuDNN Version: {torch.backends.cudnn.version() if has_cuda else "N/A"}'); \
#     print(f'Device Count: {torch.cuda.device_count() if has_cuda else "N/A"}'); \
#     try: \
#     if has_cuda and torch.cuda.device_count() > 0: \
#     print(f'Current Device: {torch.cuda.current_device()}'); \
#     print(f'Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}'); \
#     except Exception as e: print(f'Could not get device info: {e}')"

# RUN rm -rf /opt/pytorch

CMD ["/usr/local/bin/python3.9"]
