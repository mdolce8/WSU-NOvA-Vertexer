#!/usr/bin/env python
# coding: utf-8

# testing of the joint XYZ training together.

# To run this training script:
#  $ $PY37 xyz_vertex_training_testsize.py --data_train_path --epochs <200>

# ML Vtx utils
import utils.iomanager as io
import utils.model
import utils.plot
import utils.data_processing as dp

import argparse
from datetime import date
import numpy as np
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
train_path_dir = train_path.split('/')[-1]
det, horn, flux = io.IOManager.get_det_horn_and_flux_from_string(train_path_dir)

print('WARNING: You are doing testing, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

# convert the lists to numpy arrays
print('========================================')
print('Converting lists to numpy arrays...')
datasets = {key: np.array(datasets[key]) for key in datasets}

# trust that they are all numpy.arrays AND have the right shape
for key in datasets:
    print(key)
    dp.Debug(datasets[key]).printout_type()

print('========================================')
print('Addressing the bug in the h5 files. Converting firstcellx, firstcelly, firstplane to int type...')
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
print('Format of the arrays: (file_idx, event_idx, pixel_idx)......')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we set file_idx manually
print('cvnmap.shape: ', datasets['cvnmap'].shape)
print('vtx_x.shape: ', datasets['vtx.x'].shape)
print('vtx_x_pixelmap.shape: ', vtx_x_pixelmap.shape)
print('firstcellx.shape: ', datasets['firstcellx'].shape)
print('-------------------')

# this array should be something like: (2, 10000, 16000)
cvnmap = datasets['cvnmap']
print('-------------------')
print('to access FILE, index in array is: (cvnmap.shape[0]) = ', cvnmap.shape[0], 'files.')  # first dimension is the file
print('to access EVENT, index in array is: (cvnmap.shape[1]) = ', cvnmap.shape[1], 'events')  # second dimension is the events
print('to access PIXELS, index in array is: (cvnmap.shape[2]) = ', cvnmap.shape[2], 'pixels')  # third dimension is the pixelmap
print('-------------------')

print('========================================')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')

# NOTE: this still only works if the events are the same for each file...
cmap = cvnmap.reshape(total_events, 16000)
print(cmap.shape)
cmap = cmap.reshape(total_events, 100, 80, 2)  # divide into the pixels into the map
cmap = cmap.reshape(cmap.shape[0], cmap.shape[1], cmap.shape[2], cmap.shape[3], 1) # add the color channel
cvnmap_xz = cmap[:, :, :, 0].reshape(cmap.shape[0], 100, 80, 1)  # Extract the XZ view and reshape
cvnmap_yz = cmap[:, :, :, 1].reshape(cmap.shape[0], 100, 80, 1)  # Extract the YZ view and reshape
print('XZ: ', cvnmap_xz.shape)
print('YZ: ', cvnmap_yz.shape)

# TODO: this is prob how we want to do it for the 8 files (of diff. events)
# maps = []
# for file in range(len(cvnmap[0].shape)):
#     print('file: ', file)
#     cnvmap_per_file = cvnmap[file]
#     print('cnvmap_per_file: ', cnvmap_per_file.shape)
#     maps = cnvmap_per_file.reshape(file, 100, 80, 2)
#     cvnmap += cnvmap_per_file

print('========================================')
# must re-shape the vtx_x_pixelmap array to match the cvnmap_resh_xz array
# I.e. --> remove the first dimension [0], the file.
print('Reshape vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap to be of just the events...')
print('Before, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)
vtx_x_pixelmap = vtx_x_pixelmap.reshape(total_events)
vtx_y_pixelmap = vtx_y_pixelmap.reshape(total_events)
vtx_z_pixelmap = vtx_z_pixelmap.reshape(total_events)
print('After, vtx_x_pixelmap: ', vtx_x_pixelmap.shape)

vtx_coords = np.stack((vtx_x_pixelmap, vtx_y_pixelmap, vtx_z_pixelmap), axis=-1)
print('vtx_coords: ', vtx_coords.shape)

# #### Prepare the Training & Test Sets
# split the data into training (+ val) and testing sets

# XZ view and YZ view. Train on both views; predict all 3 coordinates.
# Train -- for fit(). Val -- for fit(). Test -- for evaluate()
data_train, data_val, data_test = utils.model.Config.create_test_train_val_datasets(cvnmap_xz, cvnmap_yz, vtx_coords)

# TODO: another thing to try....normalize the cvnmaps
# cvnmap_train_xz = cvnmap_train_xz / 255.0
# cvnmap_test_xz = cvnmap_test_xz / 255.0

print('========================================')
utils.model.Hardware.check_gpu_status()
print('========================================')
print('Final printout of shape before feeding into network......')
print('training: (after final reshaping)')
print("data_train['xz'].shape: ", data_train['xz'].shape)
print("data_train['yz'].shape: ", data_train['yz'].shape)
print('testing:')
print("test_train['xz'].shape: ", data_test['xz'].shape)
print("data_test['xz'].shape: ", data_test['xz'].shape)
print('========================================')

# ### MultiView Fully Connected Layer Regression CNN Model

print('using only XZ model for `x`, `y`, and `z` coordinates......')
# define separate models for each view
model_xz = utils.model.Config.create_conv2d_branch_model_single_view()
model_yz = utils.model.Config.create_conv2d_branch_model_single_view()

# create final model, join the XZ and YZ, add dense layers, and concat to 3.
model_regCNN = utils.model.Config.assemble_model_output(model_xz, model_yz)

# compile the concatenated model
model_regCNN.compile(loss='logcosh',
                     optimizer='adam',
                     metrics=['mse'])  # loss was 'mse' then 'mae'
# print a summary of the model
print(model_regCNN.summary())


# xyz-coordinate system.
date = date.today()
start = time.time()
history = model_regCNN.fit(
    x={'data_train_xz': data_train['xz'], 'data_train_yz': data_train['yz']},
    y=data_train['vtx'],
    epochs=args.epochs,
    batch_size=32,
    verbose=1,
    validation_data=({'data_val_xz': data_val['xz'], 'data_val_yz': data_val['yz']}, data_val['vtx']))
stop = time.time()
print('Time to train: ', stop - start)

# save the history to a pandas dataframe, will append the evaluate() output to this dataframe
metrics = pd.DataFrame(history.history)

# the default output name
output_name = 'testsize_{}epochs_{}_{}_{}_{}_XYZ'.format(args.epochs, det, horn, flux, date)

# save the model
save_model_dir = '/home/k948d562/output/trained-models/'
model_regCNN.save(save_model_dir + 'model_{}.h5'.format(output_name))
print('saved model to: ', save_model_dir + 'model_{}.h5'.format(output_name))
# Items in the model file: <KeysViewHDF5 ['model_weights', 'optimizer_weights']>


# Evaluate the test set
start_eval = time.time()
print('Evaluation on the test set...')
evaluation = model_regCNN.evaluate(
    x={'data_test_xz': data_test['xz'], 'data_test_yz':data_test['xz']},
    y=data_test['vtx'],
    verbose=1)
stop_eval = time.time()
print('Test Set Evaluation: {}'.format(evaluation))
print('Evaluation time: ', stop_eval - start_eval)


print('METRICS:')
print(metrics.head())
save_metric_dir = '/home/k948d562/output/metrics/'
metrics.to_csv(save_metric_dir + '/metrics_{}.csv'.format(output_name), index_label='epoch')
print('writing evaluation() to txt file...')
print('saved metrics to: ', save_metric_dir + '/metrics_{}.csv'.format(output_name))
# NOTE: evaluation only returns ONE number for each metric , and one for the loss, so just write to txt file.
with open(f'{save_metric_dir}/evaluation_results.txt', 'w') as f:
    f.write(f"Test Loss: {evaluation[0]}\n")
    f.write(f"Test MSE: {evaluation[1]}\n")
print('save evaluation to: ', save_metric_dir + '/evaluation_results.txt')


plot_dir = '/home/k948d562/plots/ml-vertexing-plots/small-scale-testing/'
utils.plot.plot_training_metrics(history, plot_dir, 'train_metrics_' + output_name)
print('Done.')