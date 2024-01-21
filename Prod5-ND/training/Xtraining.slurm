#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --job-name=XcordFHC

#SBATCH --time=24:00:00 # time

#SBATCH --ntasks=16 

#SBATCH --nodes=2 # number of nodes

#SBATCH --mem-per-cpu=40G # memory per CPU core

#SBATCH --gres=gpu:1 # request 1 gpu

#SBATCH --output=/home/m962g264/wsu_Nova_Vertexer/Source/slurm_scripts_and_outputs/s_output/%j.out

###SBATCH -p wsu_gen_highmem.q

###SBATCH --gres=killable:0

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source /home/m962g264/VirtualTensorflow/VirtualTensor/bin/activate

python /home/m962g264/wsu_Nova_Vertexer/Source/training/x-pv-finder-model-regcnn-abdulWasit.py #run python script
