#!/bin/bash
workload_port=${workload_port:-7008}

mkdir -p /data/${workload_id}
cp -n ./notebooks/*.ipynb /data/${workload_id}/

CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python3 ./startJupyter.py