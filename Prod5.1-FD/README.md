# Far Detector Prod5.1 Vertexing
### Instructions and workflow of the FD Prod5.1 Vertexing.

--- 

_Original Author_: Michael Dolce mdolce@fnal.gov

Updated: Nov. 2023

--- 

## Workflow
0. [Pre-processing](#pre-processing)
1. [Inital-checks](#initial-checks)
2. [Training](#training)
3. [Prediction](#prediction)
4. [Analysis](#analysis)


### NOTE: changing WSUID paths
The paths in the scripts are set to my WSUID.
You **MUST** change these paths to your WSUID or else you will not be able to run the scripts.
Check where they are by doing:
```grep -i "k948d562" * -r```.

## 0. Pre-processing
As it stands, the code is set up to run on the WSU cluster.
The objective is to slim the file sizeby reducing the other unnecessary NOvA variables. 
The variables we require are:

- `cvnmap` 
- `vtx.{x,y,z}`
- `firstcell{x,y}`
- `firstplane`

The `cvnmap` and `vtx.{x,y,z}` values are explicitly required for training.
The pixelmaps (`cvnmap`) are our features and the true vertex location is our label.

There are other variables we need that record the information of which cell (for x and y coordinates) and plane (for z) is the start of the pixel map. This will allow us to translate back and forth from detector space to pixel map space.  

To start, we run a bash script to create the slurm configuration for a WSU Beoshock cluster job(s). In an SSH session on the BeoShock cluster, we use this incantation:
```
. create_slurm_script_preprocess.sh  <det> <horn> <flux> <file_number>
```
where `det` is "FD" for Far Detector, `horn` is the magnetic horn current `FHC` or `RHC` (for "Forward Horn Current" and "Reverse Horn Current"), `flux` is either `nonswap` or `fluxswap` (for muon neutrinos or electron neutrinos, respectively), and `file_number` is the specific file number you want to pre-process (the h5 training files are numbered).


This script will set up the input files and output files as well as other configuration setting for a slurm job. **Be sure to change the paths of these files to your WSUID, and replace the existing one, otherwise you will not be able to proceed** (Beoshock does not allow  access to other users directories).

This script will immediately create a `submit_slurm_<some-descriptions>.sh` script. Now you can submit pre-processing to the WSU grid...

```
sbatch submit_slurm_<some-descriptions>.sh
```
 
This pre-processing can take up to 24 hours (and maybe even more). The expectation (from Oct. 2023) is that the Prod5.1 h5 files from the #reco-conveners should be slimmed from ~800MB --> ~200MB. 

For those that are curious, the `submit_slurm_<some-descriptions>.sh` script is performing this task:

```
/home/k948d562/virtual-envs/VirtualTensorFlow-Abdul/VirtualTensor/bin/python /home/k948d562/ml-vertexing/wsu-vertexer/preprocess/preprocess_h5_file.py  $PREPROCESS_FILE_PATH
```

With the completed files, training can be performed. 

But first, we want to run some initial checks on the files to make sure they are good to go.
The next section...


---

## 1. Initial-checks

At this stage, we want to make sure the information within the pre-processed files are correct and ready to be used.

Because the labels fed into our network are locations -- in detector coordinates -- and the pixel maps (used for the training),
-- the features -- are used to train the network, we need to convert the vertex location into the pixel map space. 
This step confirms everything looks OK.

To run the initial checks, we use two notebooks:
* `validate_preprocessed_h5.ipynb`
* `plot_cvnmaps.ipynb`

The first notebook is very simple, it just checks the size of the `cvnmap` (the pixel maps) and plots the first 20 vertex locations for the X coordinate.

The second notebook is more thorough. It does the following:
1. reads in the pre-processed h5 file as `numpy.ndarray`
2. checks the type and shape of the data (cvnmaps, vtx locations, and cells/plane)
3. creates a new array to convert each vertex coordinate into a pixel map coordinate (it also loads the array in a specific way because the original h5s were saved as `unsigned int` and we need to convert them to `int`).
4. lastly, it makes plots of:
   1. random pixel maps with the true vertex overlaid in yellow 'x'
   2. all the true detector vertex locations.
   3. all the true vertex locations in the 3D pixel map space

Especially for plot **1.**, it is important the vertex location is sensible, otherwise the training will produce nonsense.

Here is an example:

![example of a FD interaction of the pixel map wih the true vertex location overlaid.](https://github.com/mdolce8/WSU-NOvA-Vertexer/blob/main/Prod5.1-FD/initial-checks/cvnmap_example.png?raw=true "CVN Map example")


---


## 2. Training

Next we produce training models for each coordinate.
Again, we perform the training on the WSU cluster using slurm.
We use `training/create_slurm_script_training.sh` to create the slurm script for training.
Its arguments are:
```
COORDINATE=$1 # x, y, or z
DET=$2        # ND or FD
HORN=$3       # FHC or RHC
FLUX=$4       # Nonswap, Fluxswap
```
 which are identical arguments to the pre-processing script. We run the script like this: 

``` . create_slurm_script_training.sh <coordinate> <det> <horn> <flux>```

Before running, we must address the file path: 
* **the user must manually copy the `preprocessed_*.h5` files to the `training` directory.**
For example for the FD FHC Fluxswap sample, the preprocessed files must be copied here,
```/home/{WSUID}/output/wsu-vertexer/training/FD-Nominal-FHC-Fluxswap/```
as this is path that the training script will look for the files.
(Note the user will need to change the path to their WSUID).

Once this is addressed, we can submit the slurm script that is created from the bash script:

```sbatch submit_slurm_training_${COORDINATE}_${DET}_${HORN}_${FLUX}_${DATE}.sh```

To check on the status of the job, we can do: ```squeue -u $USER``` or `kstat` to see the status of the job.
Note, that there are `.log` and `.err` files that get made (with the same name as the sbatch script) that can be used to debug the job too (very helpful).
The job _should_ take about 24 (likely more) hours to complete. 

Once completed, the `.log` file contains useful printout information about the network, loss, and time to train, in addition,
to the names and directories of the saved models, and metrics file.
So, it is important to keep track of the `.log` file for this info.

** We note that we use a "small scale" python script to train a small subset of the train data to ensure the
plumbing is working properly. This can also be helpful for debugging. 
This script is `x_vertex_training_testsize.py`.** 


--- 

# 3. Prediction

Next, we want to use the model file to make predictions on the designated inference data set.

NOTE: 
* the inference data set is different from the training data set.
* In the FD Prod5.1 {FHC,RHC} {nonswap,fluxswap} samples, the inference data set is file 24 and/or 27.

We run the prediction with, for example, the x coordinate, 

``` python model-prediction/x_model_prediction_generation.py <det> <horn> <flux>```

where `det` is "FD" for Far Detector, `horn` is the magnetic horn current `FHC` or `RHC`
(for "Forward Horn Current" and "Reverse Horn Current"), `flux` is either `nonswap` or `fluxswap` (for muon neutrinos or electron neutrinos, respectively).

This will generate a prediction of the x vertex using the trained model.

---
IMPORTANT NOTE: there is a bug in the model that produces 256 predictions for a single vertex,
which is not exactly what we want. Instead, we want just one prediction for each vertex, so at the moment, this
code takes the mean of the 256 predictions. This will be addressed in a future PR.
---

These predictions will be saved as a CSV file that contain the following information:
```['True X', 'Reco X', 'Model Prediction']``` ordered and labeled in this fashion.

At this stage, the use can now make numerous comparisons between the model and the standard NOvA "Elastic Arms" algorithm!

Have fun!

--- 

# 4. Analysis
