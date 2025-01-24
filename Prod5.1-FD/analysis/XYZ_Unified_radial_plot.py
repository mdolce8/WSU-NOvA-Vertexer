#!/usr/bin/env python
# coding: utf-8
# # plot_1d_vertex_resolutions.py
# A. Yahaya 
# Jan. 2025
# 
# Makes plots to compare the true, model pred and Reco (EA) radial distance
# NOTE: the --test_file must be the SAME file used to make the predictions!

#  $ PY37 plot_1d_vertex_resolutions.py --pred_file ... --outdir ... 


# ML Vtx utils
import utils.data_processing as dp
import utils.iomanager as io
import utils.plot

import argparse
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="full path to CSV file of vertex predictions", default="", type=str)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
parser.add_argument("--test_file", help="full path to file used for testing/inference",
                    default="/home/k948d562/NOvA-shared/FD-Training-Samples/{}-Nominal-{}-{}/test/trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5",
                    type=str)
args = parser.parse_args()

args.pred_file = args.pred_file
args.test_file = args.test_file
args.outdir = args.outdir


# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
DET, HORN, FLUX = io.IOManager.get_det_horn_and_flux_from_string(args.pred_file)

# load the CSV file of the model predictions.
df = dp.ModelPrediction.load_pred_csv_file(args.pred_file)

# output directory
print('Output Directory: ', args.outdir)
OUTDIR = utils.plot.make_output_dir(args.outdir, 'radial', pred_filename_prefix)
str_det_horn = '{}_{}'.format(DET, HORN)


# for radial plots
bins_resolution = np.arange(0, 100, 1)  # 1 bin per cm


#radial Calculations
model_radial= np.sqrt(((df['Model Pred X'] - df['True X']) ** 2) + ((df['Model Pred Y'] - df['True Y']) ** 2) + ((df['Model Pred Z'] - df['True Z']) ** 2))
EA_radial = np.sqrt(((df['Reco X'] - df['True X']) ** 2) + ((df['Reco Y'] - df['True Y']) ** 2) + ((df['Reco Z'] - df['True Z']) ** 2))

# plot the (reco - true) vertex difference for both: Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    EA_radial,
    bins=bins_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    model_radial,
    bins=bins_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')

# Total number of events
total_events = len(df)

# Calculate total events within 10cm and 20cm for Elastic Arms
events_within_10cm_EA = np.sum(EA_radial <= 10)
events_within_20cm_EA = np.sum(EA_radial <= 20)

# Calculate total events within 10cm and 20cm for Model Prediction
events_within_10cm_Model = np.sum(model_radial <= 10)
events_within_20cm_Model = np.sum(model_radial <= 20)

# Calculate percentages
percent_10cm_EA = (events_within_10cm_EA / total_events) * 100
percent_20cm_EA = (events_within_20cm_EA / total_events) * 100
percent_10cm_Model = (events_within_10cm_Model / total_events) * 100
percent_20cm_Model = (events_within_20cm_Model / total_events) * 100


# Add percentages as text annotations
plt.text(50, hist_EA_all_res.max() * 0.55,
        f'Elastic Arms:\n10 cm: {percent_10cm_EA:.2f}%\n20 cm: {percent_20cm_EA:.2f}%',
        fontsize=12, color='black')

plt.text(50, hist_Model_all_res.max() * 0.35,
        f'Model Pred.:\n10 cm: {percent_10cm_Model:.2f}%\n20 cm: {percent_20cm_Model:.2f}%',
        fontsize=12, color='orange')

#labels
plt.xlabel('(Radial Distance) cm')
plt.ylabel('Events')
plt.text(0, hist_EA_all_res.max() * 0.75, '{} {}\nAll Interactions'.format(
    DET, HORN), fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/plot_{}_Allmodes_Radial_Distance.'.format(str_det_horn) + ext,
        dpi=300)
