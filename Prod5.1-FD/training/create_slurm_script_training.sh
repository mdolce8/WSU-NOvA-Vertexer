#!/bin/bash
# shellcheck disable=SC2086

# bash script to create a slurm script for Vertexing ML on WSU BeoShock cluster.
# Oct. 2023, M. Dolce
# run this script by:
#   $ . create_slurm_script_training.sh <coordinate> <det> <horn> <flux> <epochs>

#confusion:  from Adam Tygart...don't request multiple nodes if you can't guarantee your code can handle them.
# -- but sbatch config fails with just 1 node.

# NOTE: the x coordinate has been trained on: --ntasks=8, --mem-per-cpu=22G

DATE=$(date +%m-%d-%Y.%H.%M.%S)
echo "current date: " $DATE


COORDINATE=$1 # x, y, z, or xyz
DET=$2        # ND or FD
HORN=$3       # FHC or RHC
FLUX=$4       # Nonswap, Fluxswap, combined (both numu+nue)
EPOCHS=$5     # number of epochs to train for

echo "Coordinate: ${COORDINATE}"
echo "Detector: ${DET}"
echo "Horn: ${HORN}"
echo "Flux: ${FLUX}"
echo "Epochs: ${EPOCHS}"


# Some documentation (from latest sbatch jobs):
# -- Fluxswap:
# z coordinate takes: 16 tasks, 1 node, 20 GB mem-per-cpu.
# x coordinate took : 16 tasks, 1 node, 20 GB mem-per-cpu.
# y coordinate took : 16 tasks, 1 node, 20 GB mem-per-cpu.

# -- Nonswap:
# x coordinate took : 16 tasks, 1 node, 20 GB mem-per-cpu.
# y coordinate took : 16 tasks, 1 node, 20 GB mem-per-cpu.
# z coordinate...... ?

# TODO: investigate Physics GPU more...(can it be used together with high_mem...?)

outputfile=training_${COORDINATE}_${DET}_${HORN}_${FLUX}_${EPOCHS}Epochs_${DATE}

LOG_OUTDIR="/home/k948d562/output/logs/"

TRAINING_SCRIPT=${COORDINATE}_"vertex_training.py"

DATA_TRAIN_PATH="/home/k948d562/output/training/${DET}-Nominal-${HORN}-${FLUX}/"

slurm_dir="/home/k948d562/slurm-scripts/"
slurm_script="submit_slurm_${outputfile}.sh"

cat > $slurm_dir/submit_slurm_${outputfile}.sh <<EOS
#!/bin/bash

# script automatically generated at ${DATE} by create_slurm_script_training.sh.

## There is one strict rule for guaranteeing Slurm reads all of your options:
## Do not put *any* lines above your resource requests that aren't either:
##    1) blank. (no other characters)
##    2) comments (lines must begin with '#')

#Run this script by [ $ sbatch $slurm_script ]
#======================================================================================================================================
#SBATCH --job-name=${COORDINATE}_${DET}_${HORN}_${FLUX}_${EPOCHS}Epochs
#SBATCH --time=24:00:00

#SBATCH --output ${LOG_OUTDIR}/${outputfile}.out
#SBATCH --error  ${LOG_OUTDIR}/${outputfile}.err

#SBATCH --ntasks=16
#SBATCH --nodes=1         # number of nodes. Adam says sbatch requires at least 2, but has never worked for me.
#SBATCH --mem-per-cpu=20G # memory per CPU core
#SBATCH --gres=gpu:2      # request 2 gpu # after discussion with Abdul & Mat

#SBATCH --partition=wsu_gen_phys.q  # Physics own GPU. unclear how it works with 'highmem'
###SBATCH --partition=wsu_gen_highmem.q

###SBATCH --mail-type ALL
###SBATCH --mail-user michael.dolce@wichita.edu
#======================================================================================================================================

# create the apptainer
apptainer exec --nv /opt/beoshock/containers/beoshock_centos-7.9.sif /bin/bash -l <<EOF

# load modules
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source /home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/activate
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python --version

echo "INFO: appending MLVTX to PYTHONPATH"
export PYTHONPATH="\\\$PYTHONPATH:/homes/k948d562/ml-vertexing"
echo "PYTHONPATH is ... \\\$PYTHONPATH"

echo "/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/training/single-model/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS"
#run python script
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/training/single-model/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS

EOF

EOS

echo "Created: $slurm_dir/$slurm_script "
echo "logs will be written to: $LOG_OUTDIR"
echo "---------------------"