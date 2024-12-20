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

# convert to lowercase, upper, and capitalize
COORDINATE=${COORDINATE,,}
DET=${DET^^}
HORN=${HORN^}
FLUX=${FLUX^}

echo "Coordinate: ${COORDINATE}"
echo "Detector: ${DET}"
echo "Horn: ${HORN}"
echo "Flux: ${FLUX}"
echo "Epochs: ${EPOCHS}"

# TODO: reorganize the trimmed h5 files to single location

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

### better for a single "task"
#SBATCH --ntasks=1         # Single task
#SBATCH --cpus-per-task=8  # Allocate 8 CPUs for better parallel processing
#SBATCH --mem=380000M      # bc RealMemory=384896M for gpu201901              #  400G         # Set explicit total memory
#SBATCH --gres=gpu:2       # Request 2 GPUs

#SBATCH --nodelist=gpu201901  # compatible with TF 2.3.1

###SBATCH --mail-type ALL
###SBATCH --mail-user <WSU email address>
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

echo "/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/training/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS"
#run python script
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/training/${TRAINING_SCRIPT} --data_train_path ${DATA_TRAIN_PATH} --epochs $EPOCHS

EOF

# After the job finishes, log resource usage
sleep 60
echo "Job completed. Logging resource usage:"                    >> ${LOG_OUTDIR}/${outputfile}.out
echo "sacct -j \$SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList "    >> ${LOG_OUTDIR}/${outputfile}.out
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,NodeList >> ${LOG_OUTDIR}/${outputfile}.out

EOS

echo "Created: $slurm_dir/$slurm_script "
echo "logs will be written to: $LOG_OUTDIR/${outputfile}"
echo "---------------------"