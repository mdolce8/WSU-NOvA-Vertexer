# Far Detector Prod5.1 Vertexing
### Instructions and workflow of the FD Prod5.1 Vertexing.

--- 

_Original Author_: Michael Dolce mdolce@fnal.gov

Updated: Nov. 2023

--- 

## Workflow
0. [Pre-processing](#pre-processing)
1. [Training](#training)
2. [Prediction](#prediction)
3. [Analysis](#analysis)


## 0. Pre-processing
As it stands, the code is set up to run on the WSU cluster. The objective is to slim the file sizeby reducing the other unnecessary NOvA variables. The only variables we need are:

- `cvnmap` 
- `vtx.{x,y,z}`.

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



---