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


## 0. Pre-processing
As it stands, the code is set up to run on the WSU cluster.
The objective is to slim the file sizeby reducing the other unnecessary NOvA variables. 
The variables we require are:

- `cvnmap` 
- `vtx.{x,y,z}`
- `firstcell{x,y}
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



