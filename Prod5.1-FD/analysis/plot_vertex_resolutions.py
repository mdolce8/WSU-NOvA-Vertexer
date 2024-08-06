#!/usr/bin/env python
# coding: utf-8
# # plot_vertex_resolutions.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate).
# NOTE: this makes resolution plots of ONE COORDINATE.
# NOTE: the abs(resolution) uses the mean from the histogram, not the dataframe.
# NOTE: (reco-true)/true does not report RMS.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
#  $ PY37 plot_vertex_resolutions.py --pred_file <path/to/CSV/predictions/file> --verbose false
# 

# TODO: draw a line where the mean is....?
# TODO: should look into making a box plot somehow....


import argparse
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import seaborn as sns


# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="the CSV file of vertex predictions", default="", type=str)
parser.add_argument("--verbose", help="add printouts, helpful for debugging", default=False, type=bool)
args = parser.parse_args()

args.pred_file = args.pred_file
args.verbose = args.verbose
print('pred_file: {}'.format(args.pred_file))

# filename without the CSV and path.
pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
COORDINATE = ''
HORN = ''
FLUX = ''

# Load the CSV file of the model predictions.
# Get the detector...
print('Determining the detector...')
if 'FD' in pred_filename_prefix:
    print('I found `FD`.')
    DETECTOR = 'FD'
elif 'ND' in pred_filename_prefix:
    print('I found `ND`.')
    DETECTOR = 'ND'
else:
    print('ERROR. I did not find a detector to make predictions for, exiting......')
    exit()
print('DETECTOR: {}'.format(DETECTOR))

# Get coordinate...
print('Determining the coordinate...')
if '_X_' in pred_filename_prefix:
    print('I found `_X_`.')
    COORDINATE = 'x'
elif '_Y_' in pred_filename_prefix:
    print('I found `_Y_`.')
    COORDINATE = 'y'
elif '_Z_' in pred_filename_prefix:
    print('I found `_Z_`.')
    COORDINATE = 'z'
else:
    print('ERROR. I did not find a coordinate to make predictions for, exiting......')
    exit()
print('COORDINATE: {}'.format(COORDINATE))

# Get horn...
print('Determining the horn...')
if 'FHC' in pred_filename_prefix:
    print('I found `FHC`.')
    HORN = 'FHC'
elif 'RHC' in pred_filename_prefix:
    print('I found `RHC`.')
    HORN = 'RHC'
else:
    print('ERROR. I did not find a horn to make predictions for, exiting......')
    exit()
print('HORN: {}'.format(HORN))

# Get flux...
print('Determining the flux...')
if 'Fluxswap' in pred_filename_prefix:
    print('I found `Fluxswap`.')
    FLUX = 'Fluxswap'
elif 'Nonswap' in pred_filename_prefix:
    print('I found `Nonswap`.')
    FLUX = 'Nonswap'
else:
    print('ERROR. I did not find a flux to make predictions for, exiting......')
    exit()
print('FLUX: {}'.format(FLUX))

csvfile = pd.read_csv(args.pred_file,
                      sep=',',
                      dtype={'Row': int,
                             'True {}'.format(COORDINATE.upper()): float,
                             'Reco {}'.format(COORDINATE.upper()): float,
                             'Model Prediction': float
                             },
                      low_memory=False
                      )

# The 'Model Pred {}' is lower case. The rest are upper case.
csvfile.columns = ['Row',
                   'True {}'.format(COORDINATE.upper()),
                   'Reco {}'.format(COORDINATE.upper()),
                   'Model Pred {}'.format(COORDINATE)
                   ]
df_pred = csvfile['Model Pred {}'.format(COORDINATE)]

print(len(df_pred), 'predictions in file')
print(df_pred.head())


# define the path to the validation files
validation_path = '/Users/michaeldolce/Development/files/h5-files/validation/'
print(os.listdir(validation_path))


# Open the designated validation file
validation_file = 'trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5'.format(DETECTOR, HORN, FLUX)

file_h5 = h5py.File(validation_path + validation_file, 'r', )  # open the file:
file_h5.keys()  # see what is in the file
print("Number of events: ", len(file_h5['E']))  # count the number of events in the file

# make sure the number of events is the same in both files
assert len(df_pred) == len(file_h5['E']), print('{} != {}', len(df_pred), len(file_h5['E']))


# define path to save some plots (the local dir).
OUTDIR = ('/Users/michaeldolce/Desktop/ml-vertexing-plots/analysis/'
          + pred_filename_prefix
          + '/ resolution')

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    print('created dir: {}'.format(OUTDIR))
else:
    print('dir already exists: {}'.format(OUTDIR))


# create dataframe from the h5 validation file

# Create the DataFrame directly from the h5 file
df = pd.DataFrame({'E': file_h5['E'][:],
                   'PDG': file_h5['pdg'][:],
                   'Interaction': file_h5['interaction'][:],
                   'isCC': file_h5['iscc'][:],
                   'Mode': file_h5['mode'][:],
                   'NCid': file_h5['ncid'][:],
                   'FinalState': file_h5['finalstate'][:],
                   'Px': file_h5['p.px'][:],
                   'Py': file_h5['p.py'][:],
                   'Pz': file_h5['p.pz'][:],
                   'vtx.x': file_h5['vtx.x'][:],
                   'vtx.y': file_h5['vtx.y'][:],
                   'vtx.z': file_h5['vtx.z'][:],
                   'vtxEA.x': file_h5['vtxEA.x'][:],
                   'vtxEA.y': file_h5['vtxEA.y'][:],
                   'vtxEA.z': file_h5['vtxEA.z'][:],
                   })


# Add in the Model Prediction.
df = pd.concat([df, df_pred], axis=1)
if args.verbose:
    print(df.head())


# Add the following columns: absolute vertex difference between...
# -- abs(E.A. - True) vertex
# -- abs(Model - True) vertex.

print('Creating "AbsVtxDiff.EA.{}" and "AbsVtxDiff.Model.{}" column'.format(COORDINATE, COORDINATE))
vtx_diff_temp = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE)] - df['vtxEA.{}'.format(COORDINATE)]),
                             columns=['AbsVtxDiff.EA.{}'.format(COORDINATE)])
vtx_diff_temp_Model = pd.DataFrame(abs(df['vtx.{}'.format(COORDINATE)] - df['Model Pred {}'.format(COORDINATE)]),
                                   columns=['AbsVtxDiff.Model.{}'.format(COORDINATE)])

# TODO: maybe add the relative resolution values right here too....? --> (true - pred)/true
df = pd.concat([df, vtx_diff_temp, vtx_diff_temp_Model], axis=1)

# create a column of the Energy within NOvA's range/relevance -- for convenience.
E_nova = df[df['E'] < 5]
df['E NOvA'] = E_nova['E']
df.describe()

if args.verbose:
    print(df.head())  # check the head of the dataframe
    df.describe()  # describe the dataframe
    print(df.columns)  # print the columns of the dataframe
    # df.info() # get info on the dataframe



# Interaction types.

# NOTE: the information is stored within StandardRecord/SREnums.h
#   /// Neutrino interaction categories
#   enum mode_type_{
#     kUnknownMode               = -1,
#     kQE                        = 0,
#     kRes                       = 1,
#     kDIS                       = 2,
#     kCoh                       = 3,
#     kCohElastic                = 4,
#     kElectronScattering        = 5,
#     kIMDAnnihilation           = 6,
#     kInverseBetaDecay          = 7,
#     kGlashowResonance          = 8,
#     kAMNuGamma                 = 9,
#     kMEC                       = 10,
#     kDiffractive               = 11,
#     kEM                        = 12,
#     kWeakMix                   = 13
#   };

# assert are no kUnknownMode events...
assert (len(df[df['Mode'] == -1]) == 0)  # 'there are Unknown events. Stopping'

mode_names = ['QE', 'Res', 'DIS', 'Coh', 'CohElastic', 'ElectronScattering', 'IMDAnnihilation', 'InverseBetaDecay',
              'GlashowResonance', 'AMNuGamma', 'MEC', 'Diffractive', 'EM', 'WeakMix']

# define the colors for each mode (as close as possible to what they are in NOvA Style)

# colors for the Elastic Arms Reco.
mode_colors = {'QE': 'royalblue',
               'MEC': 'gold',
               'DIS': 'silver',
               'CohElastic': 'green',
               'Res': 'limegreen',
               'Coh': 'lightcoral',
               'ElectronScattering': 'purple',
               'IMDAnnihilation': 'pink',
               'InverseBetaDecay': 'chocolate',
               'GlashowResonance': 'cyan',
               'AMNuGamma': 'magenta',
               'Diffractive': 'dimgray',
               'EM': 'khaki',
               'WeakMix': 'teal'
               }

# colors for the model predictions
# these selected colors are darker shades from the Elastic Arms one.
mode_colors_Model = {'QE': 'navy',
                     'MEC': 'darkgoldenrod',
                     'DIS': 'grey',
                     'CohElastic': 'darkgreen',
                     'Res': 'forestgreen',
                     'Coh': 'firebrick',
                     'ElectronScattering': 'indigo',
                     'IMDAnnihilation': 'palevioletred',
                     'InverseBetaDecay': 'brown',
                     'GlashowResonance': 'cadetblue',
                     'AMNuGamma': 'darkmagenta',
                     'Diffractive': 'black',
                     'EM': 'olive',
                     'WeakMix': 'darkslategrey'
                     }

# list of the dataframes for each mode has both NC and CC.
df_modes = list()
for i in range(0, len(mode_names)):
    print(mode_names[i])
    df_modes.append(df[df['Mode'] == i])

# just as a simple check, print out the head of the QE events.
if args.verbose:
    print('printing out head() for QE events...')
    df_modes[1].head()
    print(df_modes[1].columns)

print('Creating plots..................')

############# Resolution plots ############
# Histograms. Define the binning here...

# for reco - true vertex. 
bins_resolution = np.arange(-40, 40, 1)  # 1 bin per cm. 

# for abs(reco - true) vertex difference
bins_abs_resolution = np.arange(0, 50, 1)  # 1 bin per cm. 

# for (reco - true)/true vertex difference
bins_relative_resolution = np.arange(-0.2, 0.2, .01)  # edges at +- 20%

############### Metrics ###############
# directly calculate the mean and RMS from the dataframe.
mean_Model = np.mean(df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])
rms_Model = np.sqrt(np.mean((df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) ** 2))

mean_EA = np.mean(df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])
rms_EA = np.sqrt(np.mean((df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) ** 2))


# All interactions
# ELASTIC ARMS, ONLY
# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_EA = plt.figure(figsize=(10, 8))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue

    df_mode = df_modes[i]
    hist_EA_all, bins_all, patches_all = plt.hist(
        df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)],
        bins=bins_resolution,
        range=(-50, 50),
        color=mode_colors[mode_names[i]],
        alpha=0.5,
        label=mode_names[i])


plt.title('Elastic Arms Vertex Resolution')
plt.xlabel('Reco. - True Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(15, 9e3, '{} {} {}\nElastic Arms\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=12)
plt.text(15, 5e3, 'Mean: {:.2f} cm\nRMS: {:.2f} cm'.format(mean_EA, rms_EA), fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()


for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + '/plot_{}_{}_{}_allmodes_Resolution_{}_ElasticArms.'.format(
            DETECTOR, HORN, FLUX, COORDINATE) + ext, dpi=300)



# Model Prediction, ONLY

# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_Model = plt.figure(figsize=(10, 8))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue

    df_mode = df_modes[i]
    hist_Model_all, bins_all, patches_all = plt.hist(
        df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)],
        bins=bins_resolution,
        range=(-50, 50),
        color=mode_colors_Model[mode_names[i]],
        alpha=0.5,
        label=mode_names[i])


plt.title('Model Prediction Resolution')
plt.xlabel('Reco. - True Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(25, 15e3, '{} {} {}\nModel Prediction\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=12)
plt.text(25, 5e3, 'Mean: {:.2f} cm\nRMS: {:.2f} cm'.format(mean_Model, rms_Model), fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_Model.savefig(
        OUTDIR + '/plot_{}_{}_{}_allmodes_Resolution_{}_ModelPred.'.format(
            DETECTOR, HORN, FLUX, COORDINATE) + ext, dpi=300)




# Abs(resolution)
# plot the abs(reco - true) vertex difference for both: Elastic Arms and Model Prediction

fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_abs, bins_EA_abs, patches_EA_abs = plt.hist(df['AbsVtxDiff.EA.{}'.format(COORDINATE)],
                                        bins=bins_abs_resolution,
                                        range=(-50, 50),
                                        color='black',
                                        alpha=0.5,
                                        label='Elastic Arms',
                                        hatch='//')

hist_Model_abs, bins_Model_abs, patches_Model_abs = plt.hist(df['AbsVtxDiff.Model.{}'.format(COORDINATE)],
                                                 bins=bins_abs_resolution,
                                                 range=(-50, 50),
                                                 color='orange',
                                                 alpha=0.5,
                                                 label='Model Pred.')

# NOTE: calculate the mean and RMS from the histogram here.

plt.xlabel('|Reco - True| Vertex [cm]')
plt.ylabel('Events')
plt.text(30, hist_EA_abs.max() * 0.6, '{} {} {}\nAll Interactions\n {} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE),
         fontsize=8)
plt.text(30, hist_EA_abs.max() * 0.45, 'Mean E.A.: {:.2f} cm\nMean Model: {:.2f} cm'.format(df['AbsVtxDiff.EA.{}'.format(COORDINATE)].mean(), df['AbsVtxDiff.Model.{}'.format(COORDINATE)].mean()), fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(OUTDIR + '/plot_{}_{}_{}_allmodes_{}_AbsResolution.'.
                           format(DETECTOR, HORN, FLUX, COORDINATE) + ext, dpi=300)




# Resolution
# plot the (reco - true) vertex difference for both: Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)],
    bins=bins_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)],
    bins=bins_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')


#Calculating the percentages of events within ranges of 10cm, 20cm and 30 cm for EA-True
ranges=[10, 20, 30]
total_events_EA = len(df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])
stats_text_EA = 'Total events: {} events\n'.format(total_events_EA)
for r in ranges:
    range_filter = (df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])[((df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) >= -r) & ((df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) <= r)]
    percent_events = len(range_filter) / total_events_EA * 100
    stats_text_EA += '±{} cm: {:.2f}% events\n'.format(r, percent_events)

#Calculating the percentages and mean of events within some ranges for model

ranges=[10, 20, 30]
total_events_model = len(df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])
stats_text_model = 'Total model events: {} events\n'.format(total_events_model)
for r in ranges:
    range_filter = (df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)])[((df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) >= -r) & ((df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) <= r)]
    percent_events = len(range_filter) / total_events_model * 100
    stats_text_model += '±{} cm: {:.2f}% events\n'.format(r, percent_events)


# plt.hist(np.clip(hist_Model_all_res, bins_Model_all_res[0], bins_Model_all_res[-1]), bins=bins_resolution, color='orange', alpha=0.5, label='Model Pred.')

plt.text(14, 4e4, 'Mean E.A.: {:.2f} cm\nRMS E.A.: {:.2f} cm'.format(mean_EA, rms_EA), fontsize=8)
plt.text(14, 2e4, 'Mean Model: {:.2f} cm\nRMS Model: {:.2f} cm'.format(mean_Model, rms_Model), fontsize=8)
plt.xlabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(-40, hist_EA_all_res.max() * 0.75, '{} {} {}\nAll Interactions\n {} coordinate'.format(
    DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/plot_{}_{}_{}_allmodes_{}_resolution.'.format(DETECTOR, HORN, FLUX, COORDINATE) + ext,
        dpi=300)





# Relative Resolution
# plot the (reco - true)/true vertex difference for both:
# Elastic Arms and Model Prediction
fig_resolution = plt.figure(figsize=(5, 3))

# NOTE: RMS of the residual is not reported.
mean_EA_all_relres = ((df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)]).mean()
mean_float_EA_all_relres = float(mean_EA_all_relres)
mean_Model_all_relres = ((df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)]).mean()
mean_float_Model_all_relres = float(mean_Model_all_relres)

hist_EA_all_relres, bins_EA_all_relres, patches_EA_all_relres = plt.hist(
    (df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
    bins=bins_relative_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_relres, bins_Model_all_relres, patches_Model_all_relres = plt.hist(
    (df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
    bins=bins_relative_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')
plt.xlabel('(Reco - True)/True {}'.format(COORDINATE))
plt.ylabel('Events')
plt.text(0.05, hist_EA_all_relres.max() * 0.7, '{} {} {}\nAll Interactions\n {} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE),
         fontsize=8)
plt.text(0.05,hist_EA_all_relres.max() * 0.5,
         f'Mean E.A.: {mean_float_EA_all_relres:.3f} cm\nMean Model: {mean_float_Model_all_relres:.3f} cm',
         fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/plot_{}_{}_{}_allmodes_{}_RelResolution.'.format(
            DETECTOR, HORN, FLUX, COORDINATE) + ext, dpi=300)


# Interaction type
# plot the abs(reco - true) resolution of the vertex for EACH INTERACTION TYPE
# for both: Elastic Arms and Model Prediction

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue
    fig_resolution_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]  # need to save the dataframe to a variable
    hist_EA, bins, patches = plt.hist(df_mode['AbsVtxDiff.EA.{}'.format(COORDINATE)],
                                      bins=bins_abs_resolution,
                                      range=(-50, 50),
                                      color=mode_colors[mode_names[i]],
                                      alpha=0.5,
                                      label='Elastic Arms',
                                      hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(df_mode['AbsVtxDiff.Model.{}'.format(COORDINATE)],
                                                     bins=bins_abs_resolution,
                                                     range=(-50, 50),
                                                     color=mode_colors_Model[mode_names[i]],
                                                     alpha=0.5,
                                                     label='Model Pred.')

    # NOTE: no RMS for these plots.
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.xlabel('|Reco.  - True| Vertex [cm]')
    plt.ylabel('Events')
    plt.text(35, hist_EA.max() * 0.6, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
    plt.text(35, hist_EA.max() * 0.45, 'Mean E.A.: {:.2f} cm\nMean Model: {:.2f} cm'.format(df_mode['AbsVtxDiff.EA.{}'.format(COORDINATE)].mean(), df_mode['AbsVtxDiff.Model.{}'.format(COORDINATE)].mean()), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    # plt.show()
    for ext in ['pdf', 'png']:
        fig_resolution_int.savefig(OUTDIR + '/plot_{}_{}_{}_{}_AbsResolution_{}.'.
                                   format(DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)


# FOR EACH INTERACTION TYPE
# plot the (reco - true) resolution of the vertex
# for both: Elastic Arms and Model Prediction
for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue

    fig_res_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]
    hist_EA, bins_EA, patches_EA = plt.hist(
        df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)],
        bins=bins_resolution,
        range=(-50, 50),
        color=mode_colors[mode_names[i]],
        alpha=0.5,
        label='Elastic Arms',
        hatch='//')

    hist_Model, bins_Model, patches_Model = plt.hist(
        df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)],
        bins=bins_resolution,
        range=(-50, 50),
        color=mode_colors_Model[mode_names[i]],
        alpha=0.5,
        label='Model Pred.')

    # Calculate the mean and RMS individually inside the loop here
    mean_int_EA = np.mean(df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)])
    mean_int_Model = np.mean(df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)])

    rms_int_EA = np.sqrt(np.mean((df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) ** 2))
    rms_int_Model = np.sqrt(np.mean((df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) ** 2))

    plt.xlabel('Reco. - True Vertex [cm]')
    plt.ylabel('Events')
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.text(20, hist_EA.max() * 0.55, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
    plt.text(20, hist_EA.max() * 0.45, 'Mean E.A.: {:.2f} cm\nMean Model: {:.2f} cm'.format(mean_int_EA, mean_int_Model), fontsize=8)
    plt.text(20, hist_EA.max() * 0.3, 'RMS E.A.: {:.2f} cm\nRMS Model: {:.2f} cm'.format(rms_int_EA, rms_int_Model), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    # plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(OUTDIR + '/plot_{}_{}_{}_{}_Resolution_{}.'.format(DETECTOR,
                                                                                  HORN,
                                                                                  FLUX,
                                                                                  COORDINATE,
                                                                                  mode_names[i]) + ext, dpi=300)


# FOR EACH INTERACTION TYPE
# plot the relative resolution (reco - true)/true
# for both: Elastic Arms and Model Prediction
for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue
    fig_res_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]
    hist_EA, bins, patches = plt.hist(
        (df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
            'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
        color=mode_colors[mode_names[i]], alpha=0.5, label='Elastic Arms', hatch='//')
    hist_Model, bins_Model, patches_Model = plt.hist(
        (df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
            'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
        color=mode_colors_Model[mode_names[i]], alpha=0.5, label='Model Pred.')

    # NOTE: we do not report the RMS of a residual here, that doesn't make sense...
    # NOTE: RMS of the residual is not reported.
    mean_EA_int_relres = float(((df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
        'vtx.{}'.format(COORDINATE)]).mean())

    mean_Model_int_relres = float(((df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
        'vtx.{}'.format(COORDINATE)]).mean())

    plt.xlabel('(Reco - True)/True Vertex {}'.format(COORDINATE))
    plt.ylabel('Events')
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.text(0.05, hist_EA.max() * 0.55, '{} {} {}\n{} coordinate'.format(DETECTOR, HORN, FLUX, COORDINATE), fontsize=8)
    plt.text(0.05, hist_EA.max() * 0.4, 'Mean E.A.: {:.3f} cm\nMean Model: {:.3f} cm'.format(mean_EA_int_relres, mean_Model_int_relres), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    # plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(OUTDIR + '/plot_{}_{}_{}_{}_RelResolution_{}.'.
                            format(DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)


