FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get upgrade -y \
    && apt-get install -y \
        python3-dev \
        libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*