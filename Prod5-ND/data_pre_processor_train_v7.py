# preprocess_h5_train.py

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
    # Terminal Arguments: input directory [1] output directory [2]
    indir = sys.argv[1]
    outdir = sys.argv[2]
    print('Process h5 files in '+indir+' to training files in '+outdir)
    files = [f for f in os.listdir(indir) if f.endswith('.h5')]
    print('There are '+str(len(files))+' files.')

# One file at a time to avoid problems with loading a bunch of pixel maps in memory
    for f in os.listdir(indir):
        print('Opening file.....{}'.format(f))

        # Definte the output name and don't recreate it
        outname = 'preprocessed_{}'.format(f)
        print('Creating file...{}'.format(outname))
        print(os.path.join(outdir, outname))
        if os.path.exists(os.path.join(outdir, outname)):
            print('File already exists. Continuing...')
            continue

        # Load the h5 file
        df_x = h5py.File(indir + '/' + f, 'r')['vtx.x']
        df_y = h5py.File(indir + '/' + f, 'r')['vtx.y']
        df_z = h5py.File(indir + '/' + f, 'r')['vtx.z']
        df_cvnmap = h5py.File(indir + '/' + f, 'r')['cvnmap']
        print('loaded the vertices and cvnmap dfs')

        # Save in an h5 with new dataset keys
        hf = h5py.File(os.path.join(outdir, outname), 'w')

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
        print('Created file number {}: {}'.format(f, outname))

print('All files created.')
