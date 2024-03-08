#!/usr/bin/env python
# coding: utf-8
# # plot_vertex_vs_energy.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate)
# as a function of the neutrino energy.
# NOTE: this makes resolution plots of ONE COORDINATE.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
#  $ PY37 plot_vertex_vs_energy.py --pred_file <path/to/CSV/predictions/file> --verbose false
# 



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
OUTDIR = '/Users/michaeldolce/Desktop/ml-vertexing-plots/analysis/' + pred_filename_prefix
VTX_V_ENERGY = 'vertex-vs-energy'
if not os.path.exists(OUTDIR + '/' + VTX_V_ENERGY):
    os.makedirs(OUTDIR + '/' + VTX_V_ENERGY)
    print('created dir: {}'.format(OUTDIR + '/' + VTX_V_ENERGY))
else:
    print('dir already exists: {}'.format(OUTDIR + '/' + VTX_V_ENERGY))


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



# # Vertex vs. Energy plots
# Plot the average "distance" between the E.A. Vertex and the True Vtx. 

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue

    fig_Enu = plt.figure(figsize=(7, 5))
    ax_VtxEnergy = fig_Enu.add_subplot(111)

    ax_VtxEnergy.scatter(df_modes[i]['E'], df_modes[i]['AbsVtxDiff.EA.{}'.format(COORDINATE)], marker='.', s=1,
                         label='Elastic Arms')
    ax_VtxEnergy.scatter(df_modes[i]['E'], df_modes[i]['AbsVtxDiff.Model.{}'.format(COORDINATE)], marker='.', s=1,
                         label='Model Pred.')

    plt.title('{} Interactions'.format(mode_names[i]))
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('Abs. Vertex {} Difference [cm]'.format(COORDINATE))
    plt.text(7.5, df_modes[i]['AbsVtxDiff.EA.{}'.format(COORDINATE)].max() * 0.95,
             'Events {}\n {} Coordinate'.format(len(df_modes[i]), COORDINATE), fontsize=12)
    plt.legend(loc='upper right')
    # plt.show()

    for ext in ['pdf', 'png']:
        fig_Enu.savefig(
            OUTDIR + '/' + VTX_V_ENERGY + '/plot_{}_{}_{}_abs_diff_EA_and_Model_vtx.{}_vs_E_{}.'.format(
                DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)



# ## Vertex vs. Energy plots (NOvA Energy)

# Plot the average "distance" between the E.A. Vertex and the True Vtx.
# make another set of plots that is within the energy range of NOvA 0-5 GeV...
VTX_V_ENERGY_NOVA = 'vertex-vs-energy-NOvA'

if not os.path.exists(OUTDIR + '/' + VTX_V_ENERGY_NOVA):
    os.makedirs(OUTDIR + '/' + VTX_V_ENERGY_NOVA)
    print('created dir: {}'.format(OUTDIR + '/' + VTX_V_ENERGY_NOVA))
else:
    print('dir already exists: {}'.format(OUTDIR + '/' + VTX_V_ENERGY_NOVA))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        if args.verbose:
            print('skipping ' + mode_names[i])
        continue

    fig_nova_E = plt.figure(figsize=(7, 5))
    ax_NOvAVtxEnergy = fig_nova_E.add_subplot(111)

    if args.verbose:
        print('length of df[{}] {}'.format(mode_names[i], len(df_modes[i])))

    ax_NOvAVtxEnergy.scatter(df_modes[i]['E NOvA'], df_modes[i]['AbsVtxDiff.EA.{}'.format(COORDINATE)], s=1, marker='.',
                             label='Elastic Arms')
    ax_NOvAVtxEnergy.scatter(df_modes[i]['E NOvA'], df_modes[i]['AbsVtxDiff.Model.{}'.format(COORDINATE)], s=1,
                             marker='.', label='Model Pred.')
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Abs. Vertex {} Difference [cm]'.format(COORDINATE))
    plt.text(2.5, df_modes[i]['AbsVtxDiff.EA.{}'.format(COORDINATE)].max() * 0.95, 'Events {}'.format(len(df_modes[i])),
             fontsize=8)
    plt.legend(loc='upper right')

    for ext in ['pdf', 'png']:
        fig_nova_E.savefig(
            OUTDIR + '/' + VTX_V_ENERGY_NOVA + '/plot_{}_{}_{}_abs_diff_vtx.{}_vs_E_NOvA_{}.'.format(
                DETECTOR, HORN, FLUX, COORDINATE, mode_names[i]) + ext, dpi=300)

print('Done.')
