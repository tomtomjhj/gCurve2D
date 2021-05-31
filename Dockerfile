# NOTE:          vvvvvv this should match host's CUDA version (check `nvidia-smi`)
FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# dev tools
RUN apt-get install -y build-essential make unzip autoconf automake libtool cmake g++ pkg-config
RUN apt-get install -y git wget curl gdb vim neovim htop tmux bash-completion

# dependencies
RUN apt-get install -y libgl1-mesa-dev libglew-dev freeglut3-dev
RUN apt-get install -y libcgal-dev

COPY . /gCurve2D
WORKDIR /gCurve2D
