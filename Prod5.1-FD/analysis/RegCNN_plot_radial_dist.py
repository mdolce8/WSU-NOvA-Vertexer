#!/usr/bin/env python
#coding: utf-8
# #plot_3D_radial.py
#A. Yahaya
#Aug. 2024

# Making a 3D plots to compare the radial distance true and E.A. reconstructed vertex with.
# NOTE: this makes resolution plots of the radial distance.
# --x_pred_file --> CSV model file to plot on the x-axis.
# --y_pred_file --> CSV model file to plot on the y-axis
# --z_pred_file --> CSV model file to plot on the z-axis

#Use file "27_of_28" for FD validation

# PY37 plot_3D_radial.py --x_pred_file <path/to/X_CSV/prediction/file> --y_pred_file <path/to/Y_CSV/prediction/file> --z_pred_file <path/to/Z_CSV/prediction/file>


#Get the detector...
def get_detector(pred_filename_prefix_xfile ):
   detector = ''
   print('Determining the detector...')
   if 'FD' in pred_filename_prefix_xfile:
       detector = 'FD'
   elif 'ND' in pred_filename_prefix_xfile:
       detector = 'ND'
   else:
       print('ERROR. I did not find a detector to make predictions for, exiting......')
       exit()
   print('DETECTOR: {}'.format(detector))
   return detector

def get_detector(pred_filename_prefix_yfile ):
   detector = ''
   print('Determining the detector...')
   if 'FD' in pred_filename_prefix_yfile:
       detector = 'FD'
   elif 'ND' in pred_filename_prefix_yfile:
       detector = 'ND'
   else:
       print('ERROR. I did not find a detector to make predictions for, exiting......')
       exit()
   print('DETECTOR: {}'.format(detector))
   return detector

def get_detector(pred_filename_prefix_zfile ):
   detector = ''
   print('Determining the detector...')
   if 'FD' in pred_filename_prefix_zfile:
       detector = 'FD'
   elif 'ND' in pred_filename_prefix_zfile:
       detector = 'ND'
   else:
       print('ERROR. I did not find a detector to make predictions for, exiting......')
       exit()
   print('DETECTOR: {}'.format(detector))
   return detector


## Get coordinate...
def get_coordintate(pred_filename_prefix):
   coordinate = ''
   print('Determining the coordinate...')
   if '_X_' in pred_filename_prefix:
       coordinate = 'x'
   elif '_Y_' in pred_filename_prefix:
       coordinate = 'y'
   elif '_Z_' in pred_filename_prefix:
       coordinate = 'z'
   else:
       print('ERROR. I did not find a coordinate to make predictions for, exiting......')
       exit()
   print('COORDINATE: {}'.format(coordinate))
   return coordinate
#
## Get horn...
def get_horn(pred_filename_prefix):
   horn = ''
   print('Determining the horn...')
   if 'FHC' in pred_filename_prefix:
       horn = 'FHC'
   elif 'RHC' in pred_filename_prefix:
       horn = 'RHC'
   else:
       print('ERROR. I did not find a horn to make predictions for, exiting......')
       exit()
   print('HORN: {}'.format(horn))
   return horn

# Get flux...
def get_flux(pred_filename_prefix):
    flux = ''
    print('Determining the flux...')
    if 'Fluxswap' in pred_filename_prefix:
        flux = 'Fluxswap'
    elif 'Nonswap' in pred_filename_prefix:
        flux = 'Nonswap'
    elif 'Combined' in pred_filename_prefix:
        flux = 'Combined'
    else:
        print('ERROR. I did not find a flux to make predictions for, exiting......')
        exit()
    print('FLUX: {}'.format(flux))
    return flux



import argparse
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# collect the arguments for this macro.
parser = argparse.ArgumentParser()
parser.add_argument("--x_pred_file", help="CSV file for x-axis", default="", type=str)
parser.add_argument("--y_pred_file", help="CSV file for y-axis", default="", type=str)
parser.add_argument("--z_pred_file", help="CSV file for y-axis", default="", type=str)

parser.add_argument('--verbose', dest='verbose', action='store_true', help='Enable Verbose')
parser.add_argument('--quiet', dest='quiet', action='store_false', help='Not verbose. Quiet')

# Handling the draw_contours flag
group_draw_contours = parser.add_mutually_exclusive_group()
group_draw_contours.add_argument("--draw_contours", dest='draw_contours', action="store_true", help="draw contours on the 2D hist")
group_draw_contours.add_argument("--no-draw_contours", dest='draw_contours', action="store_false", help="do not draw contours on the 2D hist")

args = parser.parse_args()


# Checking the draw_contours flag
if args.draw_contours:
    print("Drawing contours")
else:
    print("Not drawing contours")


# print the arguments
args.x_pred_file = args.x_pred_file
args.y_pred_file = args.y_pred_file
args.z_pred_file = args.z_pred_file
args.verbose = args.verbose
args.draw_contours = args.draw_contours
print('=' * 100)
print('x-pred file: {}'.format(args.x_pred_file))
print('y-pred file: {}'.format(args.y_pred_file))
print('z-pred file: {}'.format(args.z_pred_file))
print('verbose: {}'.format(args.verbose))
print('draw contours: {}'.format(args.draw_contours))
print('=' * 100)


# print the arguments
args.x_pred_file = args.x_pred_file
args.y_pred_file = args.y_pred_file
args.z_pred_file = args.z_pred_file
args.verbose = args.verbose
print('=' * 100)
print('x file: {}'.format(args.x_pred_file))
print('y file: {}'.format(args.y_pred_file))
print('z file: {}'.format(args.z_pred_file))
print('=' * 100)

# filename without the CSV and path.
pred_filename_prefix_xfile = args.x_pred_file.split('/')[-1].split('.')[0]
pred_filename_prefix_yfile = args.y_pred_file.split('/')[-1].split('.')[0]
pred_filename_prefix_zfile = args.z_pred_file.split('/')[-1].split('.')[0]

# Get the detector, horn, and flux from the filename.
COORDINATE_xfile = get_coordintate(pred_filename_prefix_xfile)
COORDINATE_yfile = get_coordintate(pred_filename_prefix_yfile)
COORDINATE_zfile = get_coordintate(pred_filename_prefix_zfile)

DETECTOR_xfile = get_detector(pred_filename_prefix_xfile)
DETECTOR_yfile = get_detector(pred_filename_prefix_yfile)
DETECTOR_zfile = get_detector(pred_filename_prefix_zfile)

HORN_xfile = get_horn(pred_filename_prefix_xfile)
HORN_yfile = get_horn(pred_filename_prefix_yfile)
HORN_zfile = get_horn(pred_filename_prefix_zfile)

FLUX_xfile = get_flux(pred_filename_prefix_xfile)
FLUX_yfile = get_flux(pred_filename_prefix_yfile)
FLUX_zfile = get_flux(pred_filename_prefix_zfile)

# check if the detector, horn, and flux are the same for all files.
assert DETECTOR_xfile == DETECTOR_yfile == DETECTOR_zfile, print('DETECTOR_xfile != DETECTOR_yfile != DETECTOR_zfile')
assert HORN_xfile == HORN_yfile == HORN_zfile, print('HORN_xfile != HORN_yfile != HORN_zfile')
assert FLUX_xfile == FLUX_yfile == FLUX_zfile, print('FLUX_xfile != FLUX_yfile != FLUX_zfile')
# coordinate should be different.
assert COORDINATE_xfile != COORDINATE_yfile != COORDINATE_zfile, print('COORDINATE_xfile == COORDINATE_yfile == COORDINATE_zfile')

# This will be the x-file reading
csvfile_x = pd.read_csv(args.x_pred_file,
                        sep=',',
                        dtype={'': int,
                                 'True {}'.format(COORDINATE_xfile.upper()): float,
                                 'Reco {}'.format(COORDINATE_xfile.upper()): float,
                                 'Model Prediction': float},
                        low_memory=False
                        )

# This will be the y-file reading
csvfile_y = pd.read_csv(args.y_pred_file,
                        sep=',',
                        dtype={'': int,
                               'True {}'.format(COORDINATE_yfile.upper()): float,
                               'Reco {}'.format(COORDINATE_yfile.upper()): float,
                               'Model Prediction': float},
                        low_memory=False
                        )
## This will be the z-file reading
csvfile_z = pd.read_csv(args.z_pred_file,
                        sep=',',
                        dtype={'': int,    
                               'True {}'.format(COORDINATE_zfile.upper()): float,
                               'Reco {}'.format(COORDINATE_zfile.upper()): float,
                               'Model Prediction': float},
                        low_memory=False
                        )
# The 'Model Pred {}' is lower case. The rest are upper case.
csvfile_x.columns = ['Row',
                     'True {}'.format(COORDINATE_xfile.upper()),
                     'Reco {}'.format(COORDINATE_xfile.upper()),
                     'Model Pred {}'.format(COORDINATE_xfile)
                     ]
csvfile_y.columns = ['Row',
                     'True {}'.format(COORDINATE_yfile.upper()),
                     'Reco {}'.format(COORDINATE_yfile.upper()),
                     'Model Pred {}'.format(COORDINATE_yfile)
                     ]

csvfile_z.columns = ['Row',
                     'True {}'.format(COORDINATE_zfile.upper()),
                     'Reco {}'.format(COORDINATE_zfile.upper()),
                     'Model Pred {}'.format(COORDINATE_zfile)
                     ]


df_pred_xfile = csvfile_x['Model Pred {}'.format(COORDINATE_xfile)]
df_pred_yfile = csvfile_y['Model Pred {}'.format(COORDINATE_yfile)]
df_pred_zfile = csvfile_z['Model Pred {}'.format(COORDINATE_zfile)]

assert len(df_pred_xfile) == len(df_pred_yfile) == len(df_pred_zfile), print('len(df_pred) != len(df_pred_yfile) != len(df_pred_zfile')
if args.verbose:
    print('len(df_pred_xfile) == len(df_pred_yfile) == len(df_pred_zfile)')

#Path to  validation file
validation_path = '/homes/m962g264/wsu_Nova_Vertexer/mike_wsu_edited_Vertexer/{}-Nominal-{}-{}/test/'.format(DETECTOR_xfile,
                                                                                           HORN_xfile,
                                                                                           FLUX_xfile)
if args.verbose:
    print(os.listdir(validation_path))

validation_file = 'trimmed_h5_R20-11-25-prod5.1reco.j_{}-Nominal-{}-{}_27_of_28.h5'.format(DETECTOR_xfile,
                                                                                           HORN_xfile,
                                                                                           FLUX_xfile)
file_h5 = h5py.File(validation_path + validation_file, 'r', )  # open the file:
file_h5.keys()  # see what is in the file
print("Number of events: ", len(file_h5['E']))  # count the number of events in the file

# make sure the number of events is the same in both files
assert len(df_pred_xfile) == len(df_pred_yfile) == len(df_pred_zfile)  == len(file_h5['E']), print('{} != {} != {}', len(df_pred_zfile),
                                                                                                   len(df_pred_yfile),
                                                                                                   len(df_pred_xfile),
                                                                                                   len(file_h5['E']))
print('=' * 100)
print(df_pred_xfile.head())
print('=' * 50)
print(df_pred_yfile.head())
print('=' * 50)
print(df_pred_zfile.head())
print('=' * 100)


# make sure the number of events is the same in both files
#assert len(df_pred) == len(file_h5['E']), print('{} != {}', len(df_pred), len(file_h5['E']))

# In[47]:


# define path to save some plots (the local dir).
LOCAL_PLOT_BASE_DIR = '/homes/m962g264/wsu_Nova_Vertexer/output/plots/New_resolution_plots' + '/' + pred_filename_prefix_zfile

# TODO: need to extract the flux too eventually...
PLOT_DIR = 'FD-{}-{}-'.format(HORN_zfile, FLUX_zfile)

OUTDIR = LOCAL_PLOT_BASE_DIR + '/' + PLOT_DIR
print('plot outdir is: {}'.format(OUTDIR))

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
#Including the Model Prediction
df = pd.concat([df, df_pred_xfile, df_pred_yfile, df_pred_zfile], axis=1)

# print the head of the dataframe
if args.verbose:
    print(df.keys())

print('Creating "VtxDiff.EA.{}" and "VtxDiff.Model.{}" column'.format(COORDINATE_xfile, COORDINATE_xfile))
EA_vtx_sqd_diff_x = pd.DataFrame(df['vtx.{}'.format(COORDINATE_xfile)] - df['vtxEA.{}'.format(COORDINATE_xfile)],
                             columns=['VtxDiff.EA.{}'.format(COORDINATE_xfile)])
model_vtx_sqd_diff_x = pd.DataFrame(df['vtx.{}'.format(COORDINATE_xfile)] - df['Model Pred {}'.format(COORDINATE_xfile)],
                                         columns=['VtxDiff.Model.{}'.format(COORDINATE_xfile)])
print('The shape of True-EA squared for x_coord is: ',EA_vtx_sqd_diff_x.shape )
print('The shape of True-model squared for x_coord is: ',model_vtx_sqd_diff_x.shape )



print('Creating y Coord  "VtxDiff.EA.{}" and "VtxDiff.Model.{}" column'.format(COORDINATE_yfile, COORDINATE_yfile))
EA_vtx_sqd_diff_y = pd.DataFrame(df['vtx.{}'.format(COORDINATE_yfile)] - df['vtxEA.{}'.format(COORDINATE_yfile)],
                             columns=['VtxDiff.EA.{}'.format(COORDINATE_yfile)])
model_vtx_sqd_diff_y = pd.DataFrame(df['vtx.{}'.format(COORDINATE_yfile)] - df['Model Pred {}'.format(COORDINATE_yfile)],
                                         columns=['VtxDiff.Model.{}'.format(COORDINATE_yfile)])
print('The shape of True-EA squared for y_coord is: ',EA_vtx_sqd_diff_y.shape )
print('The shape of True-model squared for y_coord is: ',model_vtx_sqd_diff_y.shape )



print('Creating z Coord. "VtxDiff.EA.{}" and "VtxDiff.Model.{}" column'.format(COORDINATE_zfile, COORDINATE_zfile))
EA_vtx_sqd_diff_z = pd.DataFrame(df['vtx.{}'.format(COORDINATE_zfile)] - df['vtxEA.{}'.format(COORDINATE_zfile)],
                             columns=['VtxDiff.EA.{}'.format(COORDINATE_zfile)])
model_vtx_sqd_diff_z = pd.DataFrame(df['vtx.{}'.format(COORDINATE_zfile)] - df['Model Pred {}'.format(COORDINATE_zfile)],
                                         columns=['VtxDiff.Model.{}'.format(COORDINATE_zfile)])
print('The shape of True-EA squared for z_coord is: ',EA_vtx_sqd_diff_z.shape )
print('The shape of True-model squared for z_coord is: ',model_vtx_sqd_diff_z.shape )

# The radial distance calculation 
model_radial_dist = np.sqrt(model_vtx_sqd_diff_x + model_vtx_sqd_diff_y + model_vtx_sqd_diff_z)
EA_radial_dist = np.sqrt(EA_vtx_sqd_diff_x + EA_vtx_sqd_diff_y + EA_vtx_sqd_diff_z)

# Print the first few rows of each array to inspect
print("First few rows of EA_radial_dist:")
print(EA_radial_dist[:5])  # Adjust the number to see more rows if needed

print("\nShape of EA_radial_dist:", EA_radial_dist.shape)

print("\nFirst few rows of model_radial_dist:")
print(model_radial_dist[:5])  # Adjust the number to see more rows if needed

print("\nShape of model_radial_dist:", model_radial_dist.shape)

print('The shape of the EA radial distance: ',EA_radial_dist.shape)  # Should output something like (N,) where N is the number of points
print('The shape of the model radial distance: ',model_radial_dist.shape)  # Should output something like (N,) where N is the number of points

if args.verbose:
    print(df.head())  # check the head of the dataframe
    df.describe()  # describe the dataframe
    print(df.columns)  # print the columns of the dataframe
    #df.info() # get info on the dataframe


# bin for the radial distance. 
bins_resolution = np.arange(0, 100, 1)  # 1 bin per cm. 
str_det_horn_flux = '{}_{}_{}'.format(DETECTOR_zfile, HORN_zfile, FLUX_zfile)

#Radial distances claculation for Elastic Arms and Model Prediction
dist_EA_all_res = np.sqrt(
    (df['vtx.{}'.format(COORDINATE_xfile)] - df['vtxEA.{}'.format(COORDINATE_xfile)])**2 + 
    (df['vtx.{}'.format(COORDINATE_yfile)] - df['vtxEA.{}'.format(COORDINATE_yfile)])**2 + 
    (df['vtx.{}'.format(COORDINATE_zfile)] - df['vtxEA.{}'.format(COORDINATE_zfile)])**2
)

dist_Model_all_res = np.sqrt(
    (df['vtx.{}'.format(COORDINATE_xfile)] - df['Model Pred {}'.format(COORDINATE_xfile)])**2 + 
    (df['vtx.{}'.format(COORDINATE_yfile)] - df['Model Pred {}'.format(COORDINATE_yfile)])**2 + 
    (df['vtx.{}'.format(COORDINATE_zfile)] - df['Model Pred {}'.format(COORDINATE_zfile)])**2
)



#Radial Resolution Plots
fig_resolution = plt.figure(figsize=(5, 3))

hist_EA_all_res, bins_EA_all_res, patches_EA_all_res = plt.hist(
    dist_EA_all_res,
    bins=bins_resolution,
    color='black',
    alpha=0.5,
    label='Elastic Arms',
    hatch='//')

hist_Model_all_res, bins_Model_all_res, patches_Model_all_res = plt.hist(
    dist_Model_all_res,
    bins=bins_resolution,
    color='orange',
    alpha=0.5,
    label='Model Pred.')

# Total number of events
total_events = len(df)

# Calculate total events within 10cm and 20cm for Elastic Arms
events_within_10cm_EA = np.sum(dist_EA_all_res <= 10)
events_within_20cm_EA = np.sum(dist_EA_all_res <= 20)

# Calculate total events within 10cm and 20cm for Model Prediction
events_within_10cm_Model = np.sum(dist_Model_all_res <= 10)
events_within_20cm_Model = np.sum(dist_Model_all_res <= 20)

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
plt.xlabel('(Reco - True) Radial distance [cm]')
plt.ylabel('Events')
plt.text(0, hist_EA_all_res.max() * 0.65, '{} {} {}\nAll Interactions and Coord'.format(
    DETECTOR_xfile, HORN_xfile, FLUX_xfile), fontsize=8)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)

# plt.show()
for ext in ['pdf', 'png']:
    fig_resolution.savefig(
        OUTDIR + '/plot_{}_{}_{}_allmodes_radial_resolution.'.format(DETECTOR_xfile, HORN_xfile, FLUX_xfile) + ext,
        dpi=300)



#For the 2D plots
# plot the resolution of the vertex for each interaction type on single plot.
fig_res_int_EA = plt.figure(figsize=(10, 8))
bins_resolution_1D = np.arange(0, 30, 3.9)

counts_EA_res, xbins_M_res, ybins_EA_res, imMEA = plt.hist2d(dist_Model_all_res,
           dist_EA_all_res,
           bins=(bins_resolution_1D, bins_resolution_1D), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
plt.plot([xbins_M_res.min(), xbins_M_res.max()], [xbins_M_res.min(), xbins_M_res.max()], color='white', linestyle='-', linewidth=2)
if args.draw_contours:
    plt.contour(counts_EA_res, extent=[xbins_M_res.min(), xbins_M_res.max(), ybins_EA_res.min(), ybins_EA_res.max()], linewidths=1, levels=3, colors='red', linestyle='--')

plt.title('Vertice Radial Distance  Resolution')
plt.xlabel('(Model Vertex Radial Dist {} [cm]'.format(FLUX_zfile))
plt.ylabel('(Elastic Vertex Radial Dist {} [cm]'.format(FLUX_zfile))
plt.text(10, 20, '{}'.format(str_det_horn_flux.replace("_", " ")), fontsize=12)
plt.text(xbins_M_res.max() * 0.35, ybins_EA_res.max() * 1.1, 'NOvA Simulation', fontsize=26, color='grey')
# plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + '/' + '/plot_{}_allmodes_Radial_Resolution.'.format(str_det_horn_flux) + ext,
        dpi=300)

#zoomed
bins_resolution_2D = np.arange(0, 50, 3.9) 
fig_res_int_EA = plt.figure(figsize=(10, 8))

counts_EA_res, xbins_M_res, ybins_EA_res, imMEA = plt.hist2d(dist_Model_all_res,
           dist_EA_all_res,
           bins=(bins_resolution_2D, bins_resolution_2D), cmap='viridis', cmin=1)
plt.colorbar(label='Events')
plt.plot([xbins_M_res.min(), xbins_M_res.max()], [xbins_M_res.min(), xbins_M_res.max()], color='white', linestyle='-', linewidth=2)
if args.draw_contours:
    plt.contour(counts_EA_res, extent=[xbins_M_res.min(), xbins_M_res.max(), ybins_EA_res.min(), ybins_EA_res.max()], linewidths=1, levels=3, colors='red', linestyle='--')

plt.title('Vertice Radial Distance  Resolution')
plt.xlabel('(Model Vertex Radial Dist {} [cm]'.format(FLUX_zfile))
plt.ylabel('(Elastic Vertex Radial Dist {} [cm]'.format(FLUX_zfile))
plt.text(10, 20, '{}'.format(str_det_horn_flux.replace("_", " ")), fontsize=12)
plt.text(xbins_M_res.max() * 0.35, ybins_EA_res.max() * 1.1, 'NOvA Simulation', fontsize=26, color='grey')
# plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.grid(color='black', linestyle='--', linewidth=0.25, axis='both')
# plt.show()

for ext in ['pdf', 'png']:
    fig_res_int_EA.savefig(
        OUTDIR + '/' + '/plot_{}_zoomed_allmodes_Radial_Resolution.'.format(str_det_horn_flux) + ext,
        dpi=300)

