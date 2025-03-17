FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    regex \
    einops \
    flask-restful \
    nltk \
    pytest \
    transformers \
    pytest_asyncio \
    pytest-cov \
    pytest_mock \
    pytest-random-order \
    sentencepiece \
    tiktoken \
    wrapt \
    zarr \
    wandb \
    datasets

# # Install NCCL tests to verify NCCL functionality
# RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests \
#     && cd /tmp/nccl-tests \
#     && make CUDA_HOME=/usr/local/cuda \
#     && cp /tmp/nccl-tests/build/all_reduce_perf /usr/local/bin/ \
#     && cp /tmp/nccl-tests/build/all_gather_perf /usr/local/bin/ \
#     && rm -rf /tmp/nccl-tests

# Install NVIDIA Apex properly
RUN pip uninstall -y apex || true
RUN git clone https://github.com/NVIDIA/apex.git /tmp/apex \
    && cd /tmp/apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
       --config-settings "--build-option=--cpp_ext" \
       --config-settings "--build-option=--cuda_ext" \
       ./ \
    && cd .. \
    && rm -rf /tmp/apex

WORKDIR /