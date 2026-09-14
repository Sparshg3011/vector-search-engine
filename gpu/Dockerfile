# pinned alternative to setup_pod.sh, for pods that let you bring your
# own image (runpod custom templates, vast.ai, brev with docker).
# build: docker build -t vecstore-gpu -f gpu/Dockerfile .
# run:   docker run --gpus all -it -v "$PWD":/workspace vecstore-gpu
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# python3-dev for pybind11's Python.h. cmake comes from pip, not apt:
# ubuntu 22.04 ships 3.22 and CMakeLists.txt needs 3.24.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git rsync \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "cmake>=3.24,<4" ninja \
        numpy pytest matplotlib h5py

# optional baselines — uncomment to bake them in. the faiss-gpu pip
# wheel situation is spotty; if faiss-gpu-cu12 fails to resolve, the
# stable route is conda-forge (package: faiss-gpu). cagra baselines can
# also run from the rapidsai container (cuvs preinstalled) instead of
# this image.
# RUN python3 -m pip install --no-cache-dir faiss-gpu-cu12
# RUN python3 -m pip install --no-cache-dir cuvs-cu12

WORKDIR /workspace
