#!/usr/bin/env python
# coding: utf-8

# testing of the joint XYZ training together.

# To run this training script:
#  $ $PY37 xyz_vertex_training.py --data_train_path --epochs <200>

# ML Vtx utils
import utils.iomanager as io
import utils.model
import utils.plot
import utils.data_processing as dp

import argparse
from datetime import date
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam  # optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import time

########### begin main script ###########

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--data_train_path", help="path to train data", type=str, required=True)
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
args = parser.parse_args()

# want the final dir, and extract its strings.
train_path = args.data_train_path
train_path_dir = os.path.basename(os.path.normpath(train_path))
det, horn, flux = io.IOManager.get_det_horn_and_flux_from_string(train_path_dir)

print('WARNING: You are doing full training, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

print('========================================')
# convert the lists to numpy arrays
datasets = dp.convert_lists_to_nparray(datasets)
print('to access FILE, index in array is: (cvnmap.shape[0] = ', datasets['cvnmap'].shape[0], 'files.)')  # first dimension is the file

# trust that they are all numpy.arrays AND have the right shape
for key in datasets:
    print(key)
    dp.Debug(datasets[key]).printout_type()

print('========================================')
datasets['firstcellx'] = dp.DataCleaning(datasets['firstcellx'], 'x').remove_unsigned_ints()
datasets['firstcelly'] = dp.DataCleaning(datasets['firstcelly'], 'y').remove_unsigned_ints()
datasets['firstplane'] = dp.DataCleaning(datasets['firstplane'], 'z').remove_unsigned_ints()
# Let's now check to make sure we don't have any gigantic numbers, from unsigned int's.
# we already know this is the common number we see if wew do this wrong...so check against it.
file_idx = 0
for i in [datasets['firstcellx'], datasets['firstcelly']]:
    event = 0
    if i[file_idx][event] > 4294967200:  # this a large number, just below the max value of an unsigned int, which should trigger
        print('i: ', i)
    event += 1

print('========================================')
print('Converting the vertex coordinates into pixelmap coordinates for the network...')
# Create the vertex info in pixel map coordinates:
# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_x_pixelmap = dp.ConvertFarDetCoords(det, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx'])
vtx_y_pixelmap = dp.ConvertFarDetCoords(det, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly'])
vtx_z_pixelmap = dp.ConvertFarDetCoords(det, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane'])
print('Done converting.')

print('========================================')
# Print out useful info about the shapes of the arrays
print('========================================')
print('Useful info about the shapes of the arrays:')
print('-------------------')
# we set file_idx manually
print('cvnmap.shape: ', datasets['cvnmap'].shape)
print('vtx_x.shape: ', datasets['vtx.x'].shape)
print('vtx_x_pixelmap.shape: ', vtx_x_pixelmap.shape)
print('firstcellx.shape: ', datasets['firstcellx'].shape)
print('-------------------')

print('========================================')
print('Remove the file index, and reshape the pixels into 2 (100,80) views: XZ and YZ.....')
# this array should be something like: [fileIdx](events_in_file, 16000)
# reshape the pixels into 2 (100,80) views: XZ and YZ.
# NOTE: this works regardless how many events are in each file...
cmap = np.concatenate([entry.reshape(-1, 16000) for entry in datasets['cvnmap']], axis=0)  # remove the file index (8,)
cmap = cmap.reshape(total_events, 100, 80, 2)  # divide into the pixels into the map
cmap = cmap.reshape(cmap.shape[0], cmap.shape[1], cmap.shape[2], cmap.shape[3], 1) # add the color channel
cmap = cmap.astype(np.float16)
cvnmap_xz = cmap[:, :, :, 0].reshape(cmap.shape[0], 100, 80, 1)  # Extract the XZ view and reshape
cvnmap_yz = cmap[:, :, :, 1].reshape(cmap.shape[0], 100, 80, 1)  # Extract the YZ view and reshape
# NOTE: another thing to try....normalize the cvnmaps
# print('cmap dtype: ', cmap.dtype)
# print('cvnmap_xz dtype: ', cvnmap_xz.dtype)
# print('cvnmap_yz dtype: ', cvnmap_yz.dtype)
# cvnmap_xz /= 255.0
# cvnmap_yz /= 255.0
print(cmap.shape)
print('XZ: ', cvnmap_xz.shape)
print('YZ: ', cvnmap_yz.shape)


print('========================================')
# must re-shape the vtx_x_pixelmap array to match the cvnmap_resh_xz array
# I.e. --> remove the first dimension [0], the file.
print('Reshape vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap to be of just the events...')
print('Before, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)
vtx_x_pixelmap = np.concatenate(vtx_x_pixelmap, axis=0)
vtx_y_pixelmap = np.concatenate(vtx_y_pixelmap, axis=0)
vtx_z_pixelmap = np.concatenate(vtx_z_pixelmap, axis=0)
print('After, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)

vtx_coords = np.stack((vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap), axis=-1)
print('vtx_coords: ', vtx_coords.shape)

# Remove the events with un-centered Y pixelmaps
# TODO: want to return to this. But just want to make sure all is working properly.
# vtx_coords, events_removed_y_cvnmap = dp.DataCleaning.remove_uncentered_y_cvnmaps(vtx_coords)

# #### Prepare the Training & Test Sets
# split the data into training (+ val) and testing sets

# XZ view and YZ view. Train on both views; predict all 3 coordinates.
# Train -- for fit(). Val -- for fit(). Test -- for evaluate()
data_train, data_val, data_test = utils.model.Config.create_test_train_val_datasets(cvnmap_xz, cvnmap_yz, vtx_coords)


print('========================================')
utils.model.Hardware.check_gpu_status()
print('========================================')
dp.print_input_data(data_train, data_test, data_val)
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using only XZ model for `x`, `y`, and `z` coordinates......')
# define separate models for each view
model_xz = utils.model.Config.create_conv2d_branch_model_single_view()
model_yz = utils.model.Config.create_conv2d_branch_model_single_view()

# create final model, join the XZ and YZ, add dense layers, and concat to 3.
model_regCNN = utils.model.Config.assemble_model_output(model_xz, model_yz)

utils.model.Config.compile_model(model_regCNN)
# print a summary of the model
print(model_regCNN.summary())

# xyz-coordinate system.
history = utils.model.train_model(model_regCNN,
                                  data_train,
                                  data_val,
                                  args.epochs,
                                  64)

# save the history to a pandas dataframe, will append the evaluate() output to this dataframe
metrics = pd.DataFrame(history.history)

# the default output name
output_name = '_{}epochs_{}_{}_{}_{}_XYZ'.format(args.epochs, det, horn, flux, date.today())

# save the model
save_model_dir = '/home/k948d562/output/trained-models/'
model_regCNN.save(save_model_dir + 'model_{}.h5'.format(output_name))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(output_name))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>

save_metric_dir = '/home/k948d562/output/metrics/'

# Evaluate the test set
print('METRICS:')
evaluation = utils.model.evaluate_model(model_regCNN,
                                        data_train,
                                        data_test,
                                        save_metric_dir)
print(metrics.head())
metrics.to_csv(save_metric_dir + '/metrics_{}.csv'.format(output_name), index_label='epoch')
print('Saved metrics to: ', save_metric_dir + '/metrics_{}.csv'.format(output_name))


plot_dir = '/home/k948d562/plots/ml-vertexing-plots/training/'
utils.plot.plot_training_metrics(history, plot_dir, 'train_metrics_' + output_name)
print('Done.')
