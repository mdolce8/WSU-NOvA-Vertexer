#!/bin/bash
# shellcheck disable=SC2086

# bash script to create a slurm script for Vertexing ML on WSU BeoShock cluster.
# Oct. 2023, M. Dolce
# run this script by:
#   $ . create_slurm_script_training.sh <coordinate> <det> <horn> <flux>

#confusion:  from Adam Tygart...don't request multiple nodes if you can't guarantee your code can handle them.
# -- but sbatch config fails with just 1 node.

DATE=$(date +%m-%d-%Y.%H.%M.%S)
echo "current date: " $DATE


COORDINATE=$1 # x, y, or z
DET=$2        # ND or FD
HORN=$3       # FHC or RHC
FLUX=$4       # Nonswap, Fluxswap

echo "Coordinate: $COORDINATE"
echo "Detector: $DET"
echo "Horn: $HORN"
echo "Flux: $FLUX"

outputfile=training_${COORDINATE}_${DET}_${HORN}_${FLUX}_${DATE}

OUTDIR="/home/k948d562/output/models/wsu-vertexer/${outputfile}"
echo "Outdir: $OUTDIR created."
mkdir ${OUTDIR}

slurm_dir="/home/k948d562/slurm-scripts/"
slurm_script="submit_slurm_${outputfile}.sh"

cat > $slurm_dir/submit_slurm_${outputfile}.sh <<EOF
#!/bin/bash

# script automatically generated at ${DATE} by create_slurm_script_training.sh.

## There is one strict rule for guaranteeing Slurm reads all of your options:
## Do not put *any* lines above your resource requests that aren't either:
##    1) blank. (no other characters)
##    2) comments (lines must begin with '#')

#Run this script by [ $ sbatch $slurm_script ]
#======================================================================================================================================
#SBATCH --job-name=${COORDINATE}_${DET}_${HORN}_${FLUX}
#SBATCH --time=24:00:00

#SBATCH --output ${OUTDIR}/${outputfile}.out
#SBATCH --error  ${OUTDIR}/${outputfile}.err

#SBATCH --ntasks=16
#SBATCH --nodes=2         # number of nodes, sbatch requires at least 2
#SBATCH --mem-per-cpu=40G # memory per CPU core
#SBATCH --gres=gpu:2      # request 2 gpu # after discussion with Abdul & Mat


###SBATCH --mail-type ALL
###SBATCH --mail-user michael.dolce@wichita.edu
#======================================================================================================================================


# load modules
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source /home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/activate
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python --version

echo "/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/wsu-vertexer/training/x-pv-finder-model-regcnn-abdulWasit.py --detector $DET --horn $HORN --flux $FLUX"
#run python script
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/wsu-vertexer/training/x-pv-finder-model-regcnn-abdulWasit.py --detector $DET --horn $HORN --flux $FLUX


EOF

echo "Created: $slurm_dir/$slurm_script "
echo "---------------------"