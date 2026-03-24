# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential curl ca-certificates \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN python3 -m pip install --upgrade pip

# PyTorch (CUDA 12.1)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Project requirements
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

RUN python3 -m pip install --upgrade setuptools wheel

# (For SMPL etc.)
RUN python3 -m pip install --no-cache-dir --no-build-isolation chumpy==0.70

# Constraints for pinning versions
COPY constraints.txt /workspace/constraints.txt

# OpenMMLab base
RUN python3 -m pip install --no-cache-dir -U openmim
RUN python3 -m mim install --no-cache-dir -c constraints.txt mmengine
RUN python3 -m mim install --no-cache-dir -c constraints.txt "mmcv==2.1.0"
RUN python3 -m mim install --no-cache-dir -c constraints.txt "mmdet==3.3.0"

# mmpose (editable install)
COPY mmpose /workspace/mmpose
RUN python3 -m pip install --no-cache-dir -c constraints.txt -r /workspace/mmpose/requirements.txt \
    && python3 -m pip install --no-cache-dir -c constraints.txt -v -e /workspace/mmpose

# mmaction2 (editable install)
COPY mmaction2 /workspace/mmaction2
RUN python3 -m pip install --no-cache-dir -c constraints.txt -v -e /workspace/mmaction2

# App assets
COPY app/ /workspace/app/
COPY config/ /workspace/config/
COPY checkpoints/ /workspace/checkpoints/

COPY run_pose_from_url.py /workspace/run_pose_from_url.py

# Entrypoint script
COPY start.sh /workspace/start.sh
EXPOSE 8000

CMD ["/workspace/start.sh"]