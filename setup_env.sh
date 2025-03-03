#!/bin/bash

unset PYTHONPATH
export PYTHONPATH=/homes/k948d562/virtual-envs/poetry1.7.0-test-py3.11-tf2.15/lib/python3.11/site-packages:$MLVTX
echo $PYTHONPATH
echo '=========================='
export LD_LIBRARY_PATH=/homes/k948d562/virtual-envs/poetry1.7.0-test-py3.11-tf2.15/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo 
echo "+ python envs established."
