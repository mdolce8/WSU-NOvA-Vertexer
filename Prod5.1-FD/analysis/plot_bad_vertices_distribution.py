#!/usr/bin/env python
#coding: utf-8


# To run this training script:
#  $ $PY37 plot_bad_vertices_distribution.py  --data_train_path  --outdir

# ML Vtx utils
import utils.iomanager as io
import utils.plot
import utils.data_processing as dp

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import h5py


########### begin main script ###########

# collect the arguments for this macro. the horn and swap options are required.
parser = argparse.ArgumentParser()
parser.add_argument("--data_train_path", help="path to train data", type=str, required=True)
parser.add_argument("--outdir", help="full path to output directory", default="", type=str)
args = parser.parse_args()
args.outdir = args.outdir

# want the final dir, and extract its strings.
train_path = args.data_train_path
train_path_dir = os.path.basename(os.path.normpath(train_path))
det, horn, flux = io.IOManager.get_det_horn_and_flux_from_string(train_path_dir)

print('WARNING: You are to use the full training dataset, be sure you have the correct path to train data! ')
print('data_train_path: ', train_path)
datasets, total_events, total_files = io.load_data(train_path, False)

# Convert all lists in datasets to NumPy arrays with dtype=object (while keeping the arrays within from each file arrays)
datasets = {key: np.array(value, dtype=object) for key, value in datasets.items()}
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

#output directory
outdir = "/homes/m962g264/RegCNN_Unified_Outputs/plots/"
os.makedirs(outdir, exist_ok=True)

#Converting Detector Coordinate to PIxelmap Coordinates
datasets['firstcellx']= dp.ConvertFarDetCoords(det, 'x').convert_fd_vtx_to_pixelmap(datasets['vtx.x'], datasets['firstcellx'])
datasets['firstcelly'] = dp.ConvertFarDetCoords(det, 'y').convert_fd_vtx_to_pixelmap(datasets['vtx.y'], datasets['firstcelly'])
datasets['firstplane']= dp.ConvertFarDetCoords(det, 'z').convert_fd_vtx_to_pixelmap(datasets['vtx.z'], datasets['firstplane'])
print('Convertion Done')

#Modifying a dictionary by renaming keys
datasets['vtx_x_pixelmap'] = datasets.pop('firstcellx')
datasets['vtx_y_pixelmap'] = datasets.pop('firstcelly')
datasets['vtx_z_pixelmap'] = datasets.pop('firstplane')

# concatenate/flatten each vertex array                                                                                                                                                      
print("Reshape datasets['vtx_x_pixelmap'], datasets['vtx_y_pixelmap'], datasets['vtx_z_pixelmap'] to be of just the events...")
print("Before, datasets['vtx_x_pixelmap']: ", datasets['vtx_x_pixelmap'].shape)
datasets['vtx_x_pixelmap'] = np.concatenate(datasets['vtx_x_pixelmap'], axis=0)
datasets['vtx_y_pixelmap'] = np.concatenate(datasets['vtx_y_pixelmap'], axis=0)
datasets['vtx_z_pixelmap'] = np.concatenate(datasets['vtx_z_pixelmap'], axis=0)
print("After, datasets['vtx_x_pixelmap']: ", datasets['vtx_x_pixelmap'].shape)


# combine for Nx3 array: [X, Y, Z]
vtx_coords = np.stack((datasets['vtx_x_pixelmap'], datasets['vtx_y_pixelmap'], datasets['vtx_z_pixelmap']), axis=-1)
print('Done converting.')
print('vtx_coords shape:', vtx_coords.shape)

# Filter events based on cvnmaps boundaries
filter_x = (vtx_coords[:, 0] >= 0) & (vtx_coords[:, 0] < 80)  # 0 <= x < 80
filter_y = (vtx_coords[:, 1] >= 0) & (vtx_coords[:, 1] < 80)  # 0 <= y < 80
filter_z = (vtx_coords[:, 2] >= 0) & (vtx_coords[:, 2] < 100)  # 0 <= z < 100

# Calculate the number of "Keep" and "Drop" events for each coordinate
keep_x = np.sum(filter_x)  # Count of events where x is in range
keep_y = np.sum(filter_y)  # Count of events where y is in range
keep_z = np.sum(filter_z)  # Count of events where z is in range

drop_x = np.sum(~filter_x)  # Count of events where x is out of range
drop_y = np.sum(~filter_y)  # Count of events where y is out of range
drop_z = np.sum(~filter_z)  # Count of events where z is out of range

# Calculate percentages of "Keep" and "Drop" events
total_events = len(vtx_coords)
keep_x_percent = (keep_x / total_events) * 100
keep_y_percent = (keep_y / total_events) * 100
keep_z_percent = (keep_z / total_events) * 100

drop_x_percent = (drop_x / total_events) * 100
drop_y_percent = (drop_y / total_events) * 100
drop_z_percent = (drop_z / total_events) * 100

# Print the results
print(f"Number of 'Keep' events for X: {keep_x} ({keep_x_percent:.2f}%)")
print(f"Number of 'Drop' events for X: {drop_x} ({drop_x_percent:.2f}%)")
print(f"Number of 'Keep' events for Y: {keep_y} ({keep_y_percent:.2f}%)")
print(f"Number of 'Drop' events for Y: {drop_y} ({drop_y_percent:.2f}%)")
print(f"Number of 'Keep' events for Z: {keep_z} ({keep_z_percent:.2f}%)")
print(f"Number of 'Drop' events for Z: {drop_z} ({drop_z_percent:.2f}%)")

# Create a DataFrame for plotting
df = pd.DataFrame({
    "Coordinate": ["X", "X", "Y", "Y", "Z", "Z"],
    "Classification": ["Keep", "Drop", "Keep", "Drop", "Keep", "Drop"],
    "Count": [keep_x, drop_x, keep_y, drop_y, keep_z, drop_z],
    "Percentage": [keep_x_percent, drop_x_percent, keep_y_percent, drop_y_percent, keep_z_percent, drop_z_percent]
})


# Plot the "Keep" and "Drop" counts on the same plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x="Coordinate", y="Count", hue="Classification", palette=["slategray", "yellowgreen"], width=0.6)

# Add percentage labels on top of the bars
for i, (count, percent) in enumerate(zip(df["Count"], df["Percentage"])):
    ax.text(i // 2, count + 0.1, f"{percent:.2f}%", ha="center", va="bottom", fontsize=9)

plt.title("RHC Fluxswap Number of Keep and Drop Events by Coordinate")
plt.xlabel("Coordinate")
plt.ylabel("Count")
plt.legend(title="Classification")
plt.tight_layout()

# Save the plot to the output directory
output_file = os.path.join(outdir, "RHC_Fluxswap_keep_drop_counts.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_file}")

# Show the plot (optional)
plt.show()
