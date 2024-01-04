#!/usr/bin/env python
# coding: utf-8
# # `x_model_prediction_generation.py `

# $PY37 --detector FD --horn FHC --flux Fluxswap

# # Generate a prediction from the x model and
# # save the results to a csv file, which also contains
# # the true and Elastic Arms reco vertex x values.

# NOTE: This does NOT have the weird numpy (8,) structure that the
# arrays do from the 8 training files.
# I.e. the shape is (events, pixelmap)

import argparse
import h5py
import numpy
import numpy as np
import os.path
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import time


def printout_type(array, verbose=False):
    for file in range(0, len(array)):  # there should be 8 files in each array
        if file == 0:
            if verbose:
                print('file: ', file)
                print('type of array: ', type(array[file]))
            assert (type(array[file]) is np.ndarray), "array must be a numpy array"
        if verbose:
            print('shape of array: ', array.shape)
            print('-------------------')
    print('All file entries for the array have been checked -- they are np.ndarray')


def convert_vtx_x_to_pixelmap(vtx_x_array, firstcellx_array, detStr):
    """
    :param vtx_x_array: `vtx.x` -- x in detector coordinates.
    :param firstcellx_array: `firstcellx` -- first x cell in pixelmap coordinates
    :param detStr: which detector (ND or FD)
    :return: x pixelmap coordinate
    """
    print('Converting x coordinate for {}...'.format(detStr))
    assert (type(vtx_x_array) is np.ndarray), "x_array must be a numpy array"
    if detStr == 'ND':
        return vtx_x_array / 3.99 - firstcellx_array + 48
    elif detStr == 'FD':
        return vtx_x_array / 3.97 - firstcellx_array + 192
    else:
        print('NO DETECTOR. No X coordinate conversion.')
        return None


def convert_vtx_y_to_pixelmap(vtx_y_array, firstcelly_array, detStr):
    """
    :param vtx_y_array: `vtx.y` -- y in detector coordinates.
    :param firstcelly_array: `firstcelly` -- first y cell in pixelmap coordinates
    :param detStr: which detector (ND or FD)
    :return: y pixelmap coordinate
    """
    print('Converting y coordinate for {}...'.format(detStr))
    assert (type(vtx_y_array) is np.ndarray), "y_array must be a numpy array"
    if detStr == 'ND':
        return vtx_y_array / 3.97 - firstcelly_array + 47
    elif detStr == 'FD':
        return vtx_y_array / 3.97 - firstcelly_array + 191
    else:
        print('NO DETECTOR. No Y coordinate conversion.')
        return None


def convert_vtx_z_to_pixelmap(vtx_z_array, firstplane_array, det):
    """
    :param vtx_z_array: `vtx.z` -- z in detector coordinates.
    :param firstplane_array: `firstplane` -- first plane in pixelmap coordinates
    :param det: which detector (ND or FD)
    :return: z pixelmap coordinate
    """
    print('Converting z coordinate for {}...'.format(det))
    assert (type(vtx_z_array) is np.ndarray), "z_array must be a numpy array"
    if det == 'ND':
        return vtx_z_array / 6.61 - firstplane_array
    elif det == 'FD':
        return vtx_z_array / 6.664 - firstplane_array
    else:
        print('NO DETECTOR. No Z coordinate conversion.')
        return None


def convert_x_pixelmap_to_fd_detector_coordinates(x_pixelmap_array, firstcellx_array):
  return (x_pixelmap_array + firstcellx_array - 192) * 3.97

def convY(y, fy):
  return (y+fy-47)*3.97

def convZ(z, fz):
  return (z+fz)*6.61

########### begin main script ###########


# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--detector", help="ND or FD", default="FD", type=str)
parser.add_argument("--horn", help="FHC or RHC", default="FHC", type=str)
parser.add_argument("--flux", help="nonswap (numu) or fluxswap (nue)", default="nonswap", type=str)
args = parser.parse_args()

# convert to the useful case
args.detector = args.detector.upper()
args.horn = args.horn.upper()
args.flux = args.flux.lower()
args.flux = args.flux.capitalize()  # capitalize the first letter
print(args.detector, args.horn, args.flux)


# Load the SavedModel
file_model = '/home/k948d562/output/wsu-vertexer/trained-models/make-predictions/model_training_20epochs_FD_FHC_Fluxswap_X.h5'
# could move this locally, but don't have the right TensorFlow version
model = load_model(file_model)

# Load the designated test file. This is file 24
# NOTE: there is only one test file for the FD validation, and it should be within the "test" directory
# ANOTHER NOTE: the test file has not been pre-processed,
# so it has all information within it! So can investigate deeper the interactions from each vertex.
#TODO: need to remove the hard-coding here...
path_validation = '/home/k948d562/NOvA-shared/FD-Training-Samples/FD-Nominal-FHC-Fluxswap/test/'
# this is the fluxswap validation file: trimmed_h5_R20-11-25-prod5.1reco.j_FD-Nominal-FHC-Fluxswap_24_of_28.h5'


# Read in the info from the validation file
count_validation_files = 0
events_total_validation = 0
cvnmap, vtx_x, vtx_z, firstcellx, firstplane = ([] for i in range(5))
vtx_x_EA = []

# Generally, this loop is only one file per "swap", but we will soon need one for each "swap" (i.e. 4 files)
for h5_filename in os.listdir(path_validation):
    if os.path.isdir(h5_filename):
        print('Skipping directory:', h5_filename)
        continue
    
    print('Processing... {} of {}'.format(count_validation_files, len(os.listdir(path_validation))), end="\r", flush=True)
    print('file: ', h5_filename)

    with h5py.File(path_validation + h5_filename, 'r') as f:

        if count_validation_files == 0:
            print('Keys in the file:', list(f.keys()))

        events_per_file_validation = len(f['run'][:])

        cvnmap.append(f['cvnmap'][:])
        vtx_x.append(f['vtx.x'][:])
        vtx_z.append(f['vtx.z'][:])
        firstcellx.append(f['firstcellx'][:])
        firstplane.append(f['firstplane'][:])
        vtx_x_EA.append(f['vtxEA.x'][:])

        events_total_validation += events_per_file_validation
        print('events in file: ', events_per_file_validation)
        count_validation_files += 1

print('total events: ', events_total_validation)

print('Validation files read successfully.')
print('Loaded {} files, and {} total events.'.format(count_validation_files, events_total_validation), flush=True)


# convert the lists to numpy arrays
print('========================================')
print('Converting lists to numpy arrays...')
cvnmap = np.array(cvnmap)
vtx_x = np.array(vtx_x)
vtx_z = np.array(vtx_z)
firstcellx = np.array(firstcellx)
firstplane = np.array(firstplane)
vtx_x_EA = np.array(vtx_x_EA)

# These arrays have shape (file (row), events (column), pixels)
# the first row, [0], is the first file read in, should print out the number of events here...
print('Number of events in first file:')
print('cvnmap[0].shape: ', cvnmap[0].shape)
print('vtx_x[0].shape: ', vtx_x[0].shape)
print('vtx_z[0].shape: ', vtx_z[0].shape)
print('firstcellx[0].shape: ', firstcellx[0].shape)
print('firstplane[0].shape: ', firstplane[0].shape)

print('========================================')
# trust that they are all numpy.arrays AND have the right shape
for i in [cvnmap, vtx_x, vtx_z, firstcellx, firstplane, vtx_x_EA]:
    ev = 0
    assert (type(i) is np.ndarray), "i must be a numpy array"
    if ev == 0:
        print('shape of array', i.shape)
    ev += 1

print('========================================')
printout_type(cvnmap)
printout_type(vtx_x)
printout_type(vtx_z)
printout_type(firstcellx)
printout_type(firstplane)
printout_type(vtx_x_EA)


print('========================================')
print('Addressing the bug in the h5 files. Converting firstcellx, firstcelly, firstplane to int type...')
############################################################################################################
# convert the cell and plane arrays to integers
# NOTE: for Prod5.1 h5 samples (made from Reco Conveners), the firstcellx, firstcelly arrays are `unsigned int`s.
#       this is incorrect. They need to be `int` type. So Erin E. discovered the solution that we use here:
#       -- first add 40 to each element in the array
#       -- then convert the array to `int` type
#       -- then subtract 40 from each element in the array
# We do this to `firstplane` as well (cast as int) although not strictly necessary.
# If you do not do this, firstcell numbers will be 4294967200, which is the max value of an unsigned int -- and wrong.
############################################################################################################

# some debugging, to be sure...
print('firstcellx[0][100] at start: ', firstcellx[0][100], type(firstcellx[0][100]))

print('firstcellx[fileIdx].shape', firstcellx[0].shape)
firstcellx += 40
firstcellx = np.array(firstcellx, dtype='int')
print('firstcellx[0][100] after conversion + 40 addition: ', firstcellx[0][100], type(firstcellx[0][100]))
firstcellx -= 40
print('firstcellx[0][1]] after conversion + 40 subtraction: ', firstcellx[0][100], type(firstcellx[0][100]))

firstplane = np.array(firstplane, dtype='int')  # not strictly necessary, Erin doesn't do it...


# Let's now check to make sure we don't have any gigantic numbers, from unsigned int's.
# we already know this is the common number we see if wew do this wrong...so check against it.
for i in [firstcellx, firstplane]:
    event = 0
    # assuming only 1 file used:
    if i[0][event] > 4294967200:  # this a large number, just below the max value of an unsigned int, which should trigger
        print('i: ', i)
    event += 1

# Some simple manipulation to see what's within the files...

print('========================================')
print('Converting the vertex coordinates into pixelmap coordinates for the network...')
# Create the vertex info in pixel map coordinates:
# convert the vertex location (detector coordinates) to pixel map coordinates
vtx_x_pixelmap = convert_vtx_x_to_pixelmap(vtx_x, firstcellx, args.detector)
vtx_z_pixelmap = convert_vtx_z_to_pixelmap(vtx_z, firstplane, args.detector)
print('Done converting.')

print('========================================')
# print out info for single event to double-check
print('some simple checks: ')
print('type of vtx_x_pixelmap:  ', type(vtx_x_pixelmap))
print('shape of vtx_x_pixelmap: ', vtx_x_pixelmap.shape)
print('-------------------')
print('Here is an example of the conversion for a single file :')
print('vtx_x[0] first 10 entries = ', vtx_x[0][:10])  # 0 for the first file read in
print('vtx_x_pixelmap[0] first 10 entries (these should NOT be 2e8 now) = ', vtx_x_pixelmap[0][:10])
print('firstcellx[0] after conversion, first 10 entries (these should NOT be 2e8 now): ', firstcellx[0][:10])
print('vtx_z[0] first 10 entries = ', vtx_z[0][:10])
print('vtx_z_pixelmap[0] first 10 entries = ', vtx_z_pixelmap[0][:10])
print('-------------------')


# Print out useful info about the shapes of the arrays
print('========================================')
print('Useful info about the shapes of the arrays:')
print('-------------------')
print('Format of the arrays: (file_idx, event_idx, pixel_idx)......')
print('the pixel_idx is for cvnmaps only')
print('-------------------')
# we set file_idx manually
print('cvnmap.shape: ', cvnmap.shape)
print('vtx_x.shape: ', vtx_x.shape)
print('vtx_x_pixelmap.shape: ', vtx_x_pixelmap.shape)
print('firstcellx.shape: ', firstcellx.shape)
print('-------------------')


print('========================================')
# reshape the pixels into 2 (100,80) views: XZ and YZ.
print('reshape the pixels into 2 (100,80) views: XZ and YZ.....')
print('we are also removing the file index from the arrays...')
b, cvnmap_resh_xz = [], []
x_temp, z_temp, vtx_x_pixelmap_resh, vtx_z_pixelmap_resh = [], [], [], []
# want to split this by the events amount in each file.
# loop over each file and event and append.
# We are basically removing the file index from the cvnmap array.
# NOTE: we are creating a list initially.
file_counter = 0
total_event_counter = 0
ev = 0
for file_counter in range(count_validation_files):
    print('Processing train cvnmap file {} of {}'.format(file_counter + 1, (len(os.listdir(path_validation)))))

    # loop through events in each file...in this current array structure, this is: cvnmap[file_counter].shape[1].
    # I.e. the second index within the file_idx.
    event_counter_in_file = 0
    print('beginning loop over N events....', cvnmap[file_counter].shape[0])
    print('About to reshape N ({}) events into (100, 80) map size...'.format(cvnmap[file_counter].shape[0]))
    assert cvnmap[file_counter].shape[1] == 100*80*2, 'must have 16000 pixels to reshape the maps!'
    for ev in range(cvnmap[file_counter].shape[0]):

        b = cvnmap[file_counter][event_counter_in_file].reshape(2, 100, 80)
        cvnmap_resh_xz.append(b[0])

        x_temp = vtx_x_pixelmap[file_counter][event_counter_in_file]
        vtx_x_pixelmap_resh.append(x_temp)

        z_temp = vtx_z_pixelmap[file_counter][event_counter_in_file]
        vtx_z_pixelmap_resh.append(z_temp)

        event_counter_in_file += 1
        ev += 1
    print('event_counter at end of event loop in single file: ', event_counter_in_file)
    print('ev at then end of loop: ', ev)
    total_event_counter += event_counter_in_file
    print('adding event_counter_in_file to total_event_counter. New total..... ', total_event_counter)
    file_counter += 1

assert count_validation_files == file_counter, "count_training_files must equal file_counter: {}".format(file_counter)
assert events_total_validation == total_event_counter, "total_events_training must equal event_counter: {} != {}".format(events_total_validation, total_event_counter)


# Convert the XZ view to np arrays
print('========================================')
print('Convert the XZ and YZ views to np arrays...')
cvnmap_resh_xz = np.array(cvnmap_resh_xz)  # xz views only
print('cvnmap_resh_xz.shape: ', cvnmap_resh_xz.shape)
print('----------------------------------------')
print('Convert the vtx_x_pixelmap_resh and vtx_z_pixelmap_resh to np arrays...(with NO file index now)')
vtx_x_pixelmap_resh = np.array(vtx_x_pixelmap_resh)
vtx_z_pixelmap_resh = np.array(vtx_z_pixelmap_resh)
print('vtx_z_pixelmap_resh.shape: ', vtx_z_pixelmap_resh.shape)
print('vtx_x_pixelmap_resh.shape', vtx_x_pixelmap_resh.shape)
print('========================================')

print('========================================')
# NOTE: only want one axis, so basically reshaping by keeping just the events & removing the file index
print('Remove the file index from the vtx and vtxEA, so to save into CSV file.....')
vtx_x = vtx_x[0]
vtx_x_EA = vtx_x_EA[0]
print('vtx_x.shape: ', vtx_x.shape)
print('vtx_x_EA.shape: ', vtx_x_EA.shape)
print('========================================')


# Reshape the cvnmap_resh_xz to be (events, 100, 80, 1)
cvnmap_resh_xz = cvnmap_resh_xz.reshape(cvnmap_resh_xz.shape[0], 100, 80, 1)
print('about to predict() with the model using `cvnmap_resh_xz`...')
print('cvnmap_resh_xz.shape: ', cvnmap_resh_xz.shape)
print('cvnmap_resh_xz.shape[0]: ', cvnmap_resh_xz.shape[0])
# Prediction of the coordinate in the pixelmap coordinates
start = time.time()
pred_pixelmap = model.predict([cvnmap_resh_xz])
print('Prediction done.')
end = time.time()
print('Time to predict: ', end - start)
print('-------------------')
print('the prediction `pred_pixelmap` type:', type(pred_pixelmap))
print('pred_pixelmap.shape ', pred_pixelmap.shape)
print('-------------------')
print('========================================')



# TEMPORARY: need to address the 256 predictions for each event.
print('========================================')
print('****DEC. 2023***** WE HAVE A PROBLEM WITH THE MODEL DESIGN -- ITS MAKES 256 PREDICTIONS'
      'DUE TO THE 2567 NEURONS AT THE OUTPUT LAYER. WE WILL ADDRESS THIS IN FUTURE TRAININGS, BUT FOR NOW,'
      'JUST TO EXAMINE THIS PREDICTION, WE WILL AVERAGE THE 256 PREDICTIONS TOGETHER TO GET 1 NUMBER....')
print('pred_pixelmap.shape[0] ', pred_pixelmap.shape[0])
for pred in range(pred_pixelmap.shape[0]):
    if pred < 3:
        print('pred_pixelmap[{}] shape: '.format(pred), pred_pixelmap[pred].shape)
        print('pred_pixelmap[{}] first 10 entries: '.format(pred), pred_pixelmap[pred][:10])
        print('-------------------')
    pred += 1
# we have 256 predictions of the vertex for each event.
# let's average them to get a single prediction for each event.
pred_avg = pred_pixelmap.mean(axis=1)
print('avgPred.shape: ', pred_avg.shape)
pred_avg = np.array(pred_avg)
print('========================================')



# Convert this prediction BACK into detector coordinates
pred_avg_detector = convert_x_pixelmap_to_fd_detector_coordinates(pred_avg, firstcellx[0])
pred_avg_detector = np.array(pred_avg_detector)




# Saving: True, EA Reco, and Model Prediction into CSV file
# NOTE: bc this has shape (1, events) we need the second index.
df_true_x_vtx = pd.DataFrame(vtx_x, columns=['True X'])

df_reco_x_vtx = pd.DataFrame(vtx_x_EA, columns=['Reco X'])

pred_det = pd.Series(pred_avg_detector.reshape(len(vtx_x), ))

print('df_true_x_vtx', type(df_true_x_vtx))
print('df_reco_x_vtx', type(df_reco_x_vtx))
print('pred_avg_detector', type(pred_avg_detector))
df_x_vtx = pd.concat([df_true_x_vtx, df_reco_x_vtx, pred_det], axis=1)
df_x_vtx.columns = ['True X', 'Reco X', 'Model Prediction']
subset_concat_df = df_x_vtx.iloc[:10, :10]

print('The first 10 rows and first 10 columns of concat are:\n', subset_concat_df)
outPath = '/home/k948d562/output/wsu-vertexer/predictions/'
# TODO: need to address the filename that gets saved
fileName = 'model_prediction_training_20epochs_FD_FHC_Fluxswap_X.csv'
df_x_vtx.to_csv(outPath + fileName, sep=',')
print('[True, Reco, Prediction] saved to csv file: {}{}'.format(outPath, fileName))
