ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.description="FastFlowLM build environment with all dependencies pre-installed"
LABEL org.opencontainers.image.source="https://github.com/ROCm/FastFlowLM"

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_PPA=""
ARG BACKPORTS=""

# Set up PPA if needed
RUN if [ -n "$UBUNTU_PPA" ]; then \
        apt update && apt install -y software-properties-common && \
        add-apt-repository -y "$UBUNTU_PPA"; \
    fi

# setup backports if needed
RUN if [ -n "$BACKPORTS" ]; then \
        echo "deb http://deb.debian.org/debian $BACKPORTS main" >> /etc/apt/sources.list; \
        apt update; \
        apt install -t $BACKPORTS -y libxrt-dev; \
    fi

# Install all build dependencies
RUN apt update && apt install -y \
    build-essential \
    cargo \
    cmake \
    debhelper-compat \
    dpkg-dev \
    fakeroot \
    git \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-dev \
    libboost-program-options-dev \
    libcurl4-openssl-dev \
    libdrm-dev \
    libfftw3-dev \
    libreadline-dev \
    libswresample-dev \
    libswscale-dev \
    libxrt-dev \
    nasm \
    ninja-build \
    patchelf \
    pkg-config \
    rustc \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
