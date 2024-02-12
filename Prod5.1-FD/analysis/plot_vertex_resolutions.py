#!/usr/bin/env python
# coding: utf-8
# # plot_vertex_resolutions.py
# M. Dolce. 
# Feb. 2024
# 
# Makes plots to compare the true and E.A. reconstructed vertex (from a specified coordinate).
# NOTE: this makes resolution plots of ONE COORDINATE.
# 
# This validation is using `file_27_of_28.h5` as Ashley indicated.
# 
#  $ PY37 plot_vertex_resolutions.py --pred_file <path/to/CSV/predictions/file>
#
# TODO:
#  --add quantitative metrics to the histograms
#  --make the histograms easier to distinguish (make model points?)


# In[9]:


import argparse
import os.path
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from subprocess import call
# import seaborn as sns



# In[42]:


# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", help="the CSV file of vertex predictions", default="", type=str)
args = parser.parse_args()

args.pred_file = args.pred_file
print('pred_file: {}'.format(args.pred_file))

pred_filename_prefix = args.pred_file.split('/')[-1].split('.')[0]  # get the filename only and remove the .csv
COORDINATE = ''
HORN = ''

# Load the CSV file of the model predictions.
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

# In[44]:


# define the path to the validation files
validation_path = '/Users/michaeldolce/Development/files/h5-files/validation/'
call(' ls -rtlh  $validation_path')

# TODO: will need to change the validation file in future to include Nonswap and FHC/RHC...
# Open the designated validation file
validation_file = 'trimmed_h5_R20-11-25-prod5.1reco.j_FD-Nominal-FHC-Fluxswap_27_of_28.h5'

file_h5 = h5py.File(validation_path + validation_file, 'r', )  # open the file:
file_h5.keys()  # see what is in the file
print("Number of events: ", len(file_h5['E']))  # count the number of events in the file

# make sure the number of events is the same in both files
assert len(df_pred) == len(file_h5['E']), print('{} != {}', len(df_pred), len(file_h5['E']))

# In[47]:


# define path to save some plots (the local dir).
LOCAL_PLOT_BASE_DIR = '/Users/michaeldolce/Desktop/ml-vertexing-plots/fd-validation' + '/' + pred_filename_prefix

# TODO: need to extract the flux too eventually...
PLOT_DIR = 'FD-{}-Fluxswap-{}'.format(HORN, COORDINATE.upper())

OUTDIR = LOCAL_PLOT_BASE_DIR + '/' + PLOT_DIR
print('plot outdir is: {}'.format(OUTDIR))

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    print('created dir: {}'.format(OUTDIR))
else:
    print('dir already exists: {}'.format(OUTDIR))

# In[48]:


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

# In[49]:

# Add in the Model Prediction.
df = pd.concat([df, df_pred], axis=1)
print(df.head())

# In[50]:


# Add the following columns: absolute vertex difference between...
# -- abs(E.A. - True) vertex
# -- abs(Model - True) vertex.

print('Creating "AbsVtxDiff.EA.{}" and "AbsVtxDiff.Model{}" column'.format(COORDINATE, COORDINATE))
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

print(df.head())  # check the head of the dataframe
df.describe()  # describe the dataframe
print(df.columns)  # print the columns of the dataframe
# df.info() # get info on the dataframe


# In[53]:


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
print('printing out head() for QE events...')
df_modes[1].head()
print(df_modes[1].columns)

# In[55]:

# Vertex Location plots

# Plot the Reco vs. True vertex for the nue events broken down by interaction type

print('PLOTTING THE TRUE VS. E.A. & Model VERTEX Location FOR EACH INTERACTION TYPE......')
print('coordinate.....{}'.format(COORDINATE))
VTX_LOCATION = 'vertex-location'

if not os.path.exists(OUTDIR + '/' + VTX_LOCATION):
    os.makedirs(OUTDIR + '/' + VTX_LOCATION)
    print('created dir: {}'.format(OUTDIR + '/' + VTX_LOCATION))
else:
    print('dir already exists: {}'.format(OUTDIR + '/' + VTX_LOCATION))

for i in range(0, len(mode_names)):
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue
    fig_reco_true = plt.figure(figsize=(10, 10))
    ax_VtxLocation = fig_reco_true.add_subplot(111)
    ax_VtxLocation.scatter(df_modes[i]['vtx.{}'.format(COORDINATE)][:], df_modes[i]['vtxEA.{}'.format(COORDINATE)][:],
                           s=1, label='Elastic Arms')
    ax_VtxLocation.scatter(df_modes[i]['vtx.{}'.format(COORDINATE)][:],
                           df_modes[i]['Model Pred {}'.format(COORDINATE)][:], s=1, marker='x', color='orange',
                           label='Model Pred.')
    plt.title('{} Vertex Location'.format(mode_names[i]))
    plt.xlabel('True Vtx {} [cm]'.format(COORDINATE))
    plt.ylabel('Reco. Vtx {} [cm]'.format(COORDINATE))
    plt.plot(np.arange(-750, 750, 1), np.arange(-750, 750, 1), '--', lw=1.5, color='gray')
    plt.legend(loc='upper left')
    plt.text(550, 1000, 'FD {} Fluxswap \n {} Vertex\n Validation'.format(HORN, COORDINATE), fontsize=12)

    # plt.xlim(-150, 150)
    # plt.ylim(-150, 150)
    # plt.show()

    for ext in ['pdf', 'png']:
        fig_reco_true.savefig(
            OUTDIR + '/' + VTX_LOCATION + '/FD_{}_Fluxswap_EA_and_Model_vs_true_vtx_{}.{}.'.format(
                HORN, COORDINATE, mode_names[i]) + ext, dpi=300)

print(df_modes[1].columns)

# # Vertex vs. Energy plots

# In[56]:


# Plot the average "distance" between the E.A. Vertex and the True Vtx. 
VTX_V_ENERGY = 'vertex-vs-energy'

if not os.path.exists(OUTDIR + '/' + VTX_V_ENERGY):
    os.makedirs(OUTDIR + '/' + VTX_V_ENERGY)
    print('created dir: {}'.format(OUTDIR + '/' + VTX_V_ENERGY))
else:
    print('dir already exists: {}'.format(OUTDIR + '/' + VTX_V_ENERGY))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
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
    plt.show()

    for ext in ['pdf', 'png']:
        fig_Enu.savefig(
            OUTDIR + '/' + VTX_V_ENERGY + '/plot_FD_{}_Fluxswap_abs_diff_EA_and_Model_vtx.{}_vs_E_{}.'.format(
                HORN, COORDINATE, mode_names[i]) + ext, dpi=300)

# ## Vertex vs. Energy plots (NOvA Energy)

# In[57]:


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
        print('skipping ' + mode_names[i])
        continue

    fig_nova_E = plt.figure(figsize=(7, 5))
    ax_NOvAVtxEnergy = fig_nova_E.add_subplot(111)

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
            OUTDIR + '/' + VTX_V_ENERGY_NOVA + '/plot_FD_{}_Fluxswap_abs_diff_vtx.{}_vs_E_NOvA_{}.'.format(
                HORN, COORDINATE, mode_names[i]) + ext, dpi=300)

#  # Resolution plots
# 

# ### Define the binning here...

# In[58]:


# for reco - true vertex. 
bins_resolution = np.arange(-40, 40, 1)  # 1 bin per cm. 

# for abs(reco - true) vertex difference
bins_abs_resolution = np.arange(0, 50, 1)  # 1 bin per cm. 

# for (reco - true)/true vertex difference
bins_relative_resolution = np.arange(-0.2, 0.2, .01)  # edges at +- 20%

# ### define the plot names...

# In[59]:


RESOLUTION = 'resolution'  # outdir name

if not os.path.exists(OUTDIR + '/' + RESOLUTION):
    os.makedirs(OUTDIR + '/' + RESOLUTION)
    print('created dir: {}'.format(OUTDIR + '/' + RESOLUTION))
else:
    print('dir already exists: {}'.format(OUTDIR + '/' + RESOLUTION))

# ## All interactions

# In[60]:


# ELASTIC ARMS

# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_EA = plt.figure(figsize=(10, 8))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue

    df_mode = df_modes[i]
    hist_EA_all, bins_all, patches_all = plt.hist(
        df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
        range=(-50, 50), color=mode_colors[mode_names[i]], alpha=0.5, label=mode_names[i])

plt.title('Elastic Arms Vertex Resolution')
plt.xlabel('Reco. - True Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(25, 1e4, 'FD {} Fluxswap\nElastic Arms\n{} coordinate'.format(HORN, COORDINATE), fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_allmodes_Resolution_{}_ElatsticArms.'.format(
            HORN, COORDINATE) + ext, dpi=300)

# In[61]:


# Model Prediction

# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_Model = plt.figure(figsize=(10, 8))

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue

    df_mode = df_modes[i]
    hist_Model_all, bins_all, patches_all = plt.hist(
        df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
        range=(-50, 50), color=mode_colors_Model[mode_names[i]], alpha=0.5, label=mode_names[i])

plt.title('Model Prediction Resolution')
plt.xlabel('Reco. - True Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(25, 20, 'FD {} Fluxswap\nModel Prediction\n{} coordinate'.format(HORN, COORDINATE), fontsize=12)
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_Model.savefig(
        OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_allmodes_Resolution_{}_ModelPred.'.format(
            HORN, COORDINATE) + ext, dpi=300)

# ### Abs(resolution)

# In[62]:


# plot the abs(reco - true) vertex difference for all events

fig_resolution = plt.figure(figsize=(5, 3))

hist_EA, bins_EA, patches_EA = plt.hist(df['AbsVtxDiff.EA.{}'.format(COORDINATE)], bins=bins_abs_resolution,
                                        range=(-50, 50), color='black', alpha=0.5, label='Elastic Arms')
hist_Model, bins_Model, patches_Model = plt.hist(df['AbsVtxDiff.Model.{}'.format(COORDINATE)], bins=bins_abs_resolution,
                                                 range=(-50, 50), color='orange', alpha=0.5, label='Model Pred.')
plt.xlabel('|Reco - True| Vertex [cm]')
plt.ylabel('Events')
plt.text(30, hist_EA.max() * 0.6, 'FD {} Fluxswap\nAll Interactions\n {} coordinate'.format(HORN, COORDINATE),
         fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_allmodes_{}_AbsResolution.'.
                           format(HORN,COORDINATE) + ext, dpi=300)

# ### Resolution

# In[63]:


# plot the (reco - true) vertex difference for all events

fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)], bins=bins_resolution, color='black', alpha=0.5,
    label='Elastic Arms')
hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)], bins=bins_resolution, color='orange',
    alpha=0.5, label='Model Pred.')
# plt.hist(np.clip(hist_Model_all_res, bins_Model_all_res[0], bins_Model_all_res[-1]), bins=bins_resolution, color='orange', alpha=0.5, label='Model Pred.')
plt.xlabel('(Reco - True) Vertex {} [cm]'.format(COORDINATE))
plt.ylabel('Events')
plt.text(-40, hist_EA_all_res.max() * 0.75, 'FD {} Fluxswap\nAll Interactions\n {} coordinate'.format(HORN, COORDINATE),
         fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_allmodes_{}_resolution.'.format(HORN, COORDINATE) + ext,
        dpi=300)

# ### Relative Resolution

# In[64]:


# plot the (reco - true)/true vertex difference for all events

fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_relres, bins_EA_all_relres, patches_EA_all_relres = plt.hist(
    (df['vtxEA.{}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
    bins=bins_relative_resolution, color='black', alpha=0.5, label='Elastic Arms')
hist_Model_all_relres, bins_Model_all_relres, patches_Model_all_relres = plt.hist(
    (df['Model Pred {}'.format(COORDINATE)] - df['vtx.{}'.format(COORDINATE)]) / df['vtx.{}'.format(COORDINATE)],
    bins=bins_relative_resolution, color='orange', alpha=0.5, label='Model Pred.')
plt.xlabel('(Reco - True)/True [cm]')
plt.ylabel('Events')
plt.text(0.13, hist_EA.max() * 0.7, 'FD {} Fluxswap\nAll Interactions\n {} coordinate'.format(HORN, COORDINATE),
         fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_allmodes_{}_RelResolution.'.format(
            HORN, COORDINATE) + ext, dpi=300)

# ## Interaction type

# In[65]:


# plot the abs(reco - true) resolution of the vertex for each interaction type

# TODO: need to save these interaction type plots to a separate dir...!

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue
    fig_resolution_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]  # need to save the dataframe to a variable
    hist_EA, bins, patches = plt.hist(df_mode['AbsVtxDiff.EA.{}'.format(COORDINATE)], bins=bins_abs_resolution,
                                      range=(-50, 50), color=mode_colors[mode_names[i]], alpha=0.5,
                                      label='Elastic Arms')
    hist_Model, bins_Model, patches_Model = plt.hist(df_mode['AbsVtxDiff.Model.{}'.format(COORDINATE)],
                                                     bins=bins_abs_resolution, range=(-50, 50),
                                                     color=mode_colors_Model[mode_names[i]], alpha=0.5,
                                                     label='Model Pred.')
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.xlabel('|Reco.  - True| Vertex [cm]')
    plt.ylabel('Events')
    plt.text(35, hist_EA.max() * 0.6, 'FD {} Fluxswap\n{} coordinate'.format(HORN, COORDINATE), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    plt.show()
    for ext in ['pdf', 'png']:
        fig_resolution_int.savefig(OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_{}_AbsResolution_{}.'.
                                   format(HORN, COORDINATE, mode_names[i]) + ext, dpi=300)

# In[66]:


# plot the (reco - true) resolution of the vertex for each interaction type

for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue

    fig_res_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]
    hist_EA, bins_EA, patches_EA = plt.hist(
        df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
        range=(-50, 50), color=mode_colors[mode_names[i]], alpha=0.5, label='Elastic Arms')
    hist_Model, bins_Model, patches_Model = plt.hist(
        df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)], bins=bins_resolution,
        range=(-50, 50), color=mode_colors_Model[mode_names[i]], alpha=0.5, label='Model Pred.')
    plt.xlabel('Reco. - True Vertex [cm]')
    plt.ylabel('Events')
    plt.title('{} Interactions'.format(mode_names[i]))
    plt.text(20, hist_EA.max() * 0.55, 'FD {} Fluxswap\n{} coordinate'.format(HORN, COORDINATE), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(
            OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_{}_Resolution_{}.'.format(HORN,
                                                                                        COORDINATE,
                                                                                        mode_names[i]) + ext, dpi=300)

# In[67]:


# plot the relative resolution (reco - true)/true


for i in range(0, len(mode_names)):
    # ignore the empty Interaction dataframes 
    if df_modes[i].empty:
        print('skipping ' + mode_names[i])
        continue
    fig_res_int = plt.figure(figsize=(5, 3))

    df_mode = df_modes[i]
    hist_EA, bins, patches = plt.hist(
        (df_mode['vtxEA.{}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
            'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
        color=mode_colors[mode_names[i]], alpha=0.5, label='Elastic Arms')
    hist_Model, bins_Model, patches_Model = plt.hist(
        (df_mode['Model Pred {}'.format(COORDINATE)] - df_mode['vtx.{}'.format(COORDINATE)]) / df_mode[
            'vtx.{}'.format(COORDINATE)], bins=bins_relative_resolution, range=(-50, 50),
        color=mode_colors_Model[mode_names[i]], alpha=0.5, label='Model Pred.')
    plt.xlabel('(Reco - True)/True Vertex {} [cm]'.format(COORDINATE))
    plt.ylabel('Events')
    plt.text(0.1, hist_EA.max() * 0.55, 'FD {} Fluxswap\n{} coordinate'.format(HORN, COORDINATE), fontsize=8)
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')

    plt.show()
    for ext in ['pdf', 'png']:
        fig_res_int.savefig(OUTDIR + '/' + RESOLUTION + '/plot_FD_{}_Fluxswap_{}_RelResolution_{}.'.
                            format(HORN, COORDINATE, mode_names[i]) + ext, dpi=300)

# TODO: should look into making a box plot somehow....
# make the width of the box the 10% difference from the mean or something like it?
