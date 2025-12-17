# Dockerfile for work with FlashMoE on CUDA (https://github.com/osayamenja/FlashMoE.git)

# Based on CUDA docker image
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# Install dependencies
ARG DEPENDENCIES="cmake \
                  git \
                  libboost-all-dev \
                  libmpich-dev \
                  libopenmpi-dev \
                  nano \
                  ninja-build \
                  python3.12-venv \
                  wget"
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

# NVSHMEM
RUN cd /opt && wget https://github.com/NVIDIA/nvshmem/releases/download/v3.4.5-0/nvshmem_src_cuda-all-all-3.4.5.tar.gz && \
    tar xvf nvshmem_src_cuda-all-all-3.4.5.tar.gz
RUN mkdir nvshmem_build && cd nvshmem_build
RUN export CUDAFLAGS='-fpermissive' && export CXXFLAGS='-fpermissive' && \
    cmake -S/opt/nvshmem_src -B.  -Wno-dev \
        -DNVSHMEM_USE_GDRCOPY=0 \
        -DCUDA_HOME="/usr/local/cuda" \
        -DMPI_HOME="/usr/lib/x86_64-linux-gnu/openmpi" \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_EXAMPLES=OFF && \
    make -j install
RUN cd /opt/nvshmem_src/scripts &&
    ./install_hydra.sh /opt/hydra_src /opt/hydra_install

# MathDX
RUN cd /opt && wget https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz && \
    tar xvf nvidia-mathdx-25.06.1-cuda12.tar.gz

# FlashMoE
RUN cd /opt && git clone https://github.com/osayamenja/FlashMoE.git
RUN cd /opt/FlashMoE/csrc && mkdir build && cd build

# Command below should be runin the container since they require HW information
#RUN cmake -DCMAKE_BUILD_TYPE=Release \
#    -G Ninja -S/opt/FlashMoE/csrc -B. \
#    -Wno-dev -DNVSHMEM_HOME=/usr/local/nvshmem/ \
#    -Dmathdx_DIR=/opt/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/lib/cmake/mathdx
#RUN cmake --build . -j
#WORKDIR /opt
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvshmem/lib/

CMD ["/bin/bash"]

