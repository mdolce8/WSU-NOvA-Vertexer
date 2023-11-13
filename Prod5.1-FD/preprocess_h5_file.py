# $PY37 preprocess_h5_file.py <infile_h5>

# python script to preprocess the h5 files for training
# which cuts out everything but the vertex and cvnmap info
# (vtx.x, vtx.y, vtx.z, cvnmap).

# this script is submitted to the BeoShock (WSU) cluster
# where each slurm submission processes one file at a time

# M. Dolce
# Oct. 2023

import os
import sys
import h5py
import numpy as np

if __name__ == '__main__':
    # Terminal Arguments: input file [1]
    inFilePath = sys.argv[1]
    inPath = inFilePath.split('/')[:-1]
    infile = inFilePath.split('/')[-1]
    # print('input path: ', inPath)

    # this outPath is hard-coded, it's in "my" directory
    outPath = '/home/k948d562/output/wsu-vertexer/preprocess'
    outName = 'preprocessed_{}'.format(infile)
    print('Processing h5 file: ' + infile)
    print('Saving for training to ' + outPath)

    # Don't recreate the file if it exists
    print('Creating file...{}'.format(os.path.join(outPath, outName)))
    if os.path.exists(os.path.join(outPath, outName)):
        print('File already exists. Don\'t want to overwrite! Exiting...')
        exit(0)

    # Load the h5 file
    # One file at a time to avoid problems with loading a bunch of pixel maps in memory
    print('Opening file.....{}'.format(infile))
    df_x = h5py.File(inFilePath, 'r')['vtx.x']
    df_y = h5py.File(inFilePath, 'r')['vtx.y']
    df_z = h5py.File(inFilePath, 'r')['vtx.z']
    df_cvnmap = h5py.File(inFilePath, 'r')['cvnmap']
    print('loaded the vertices and cvnmap dfs')

    # Save in an h5 with new dataset keys
    hf = h5py.File(os.path.join(outPath, outName), 'w')

    hf.create_dataset('vtx.x',  data=df_x,                compression='gzip')
    print('added vtx.x')
    hf.create_dataset('vtx.y',  data=df_y,                compression='gzip')
    print('added vtx.y')
    hf.create_dataset('vtx.z',  data=df_z,                compression='gzip')
    print('added vtx.z')

    # save as 'chunks' to save space, since each pixel map is this size.
    hf.create_dataset('cvnmap', data=np.stack(df_cvnmap), chunks=(1, 16000),  compression='gzip')
    print('added cvnmap')

    hf.close()
    print('File created: ', outName)
