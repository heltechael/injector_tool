#!/bin/bash

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))

export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/root/miniconda3/pkgs/cuda-nvcc-11.8.89-0/

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
