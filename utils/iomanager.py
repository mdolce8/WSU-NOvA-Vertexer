# iomanager.py
# Tools to assist reading and writing.

import os
import h5py

def load_data(path_to_data, load_elasticarms = False):
    """
    :param path_to_data: the _complete_ path (works for training AND test/validation)
    :param load_elasticarms: include E.A. info in dataset to load
    :return: datasets dictionary (of all relevant Vars), file count, event count
    """
    datasets = {
        "cvnmap": [],
        "vtx.x": [],
        "vtx.y": [],
        "vtx.z": [],
        "firstcellx": [],
        "firstcelly": [],
        "firstplane": []
    }
    if load_elasticarms:
        print('adding E.A. info to \'datasets\'...')
        datasets["vtxEA.x"] = []
        datasets["vtxEA.y"] = []
        datasets["vtxEA.z"] = []

    total_files = 0
    total_events = 0
    # Process each file
    for h5_filename in os.listdir(path_to_data):
        if not h5_filename.endswith('.h5'):
            print('Skipping this file or dir:', h5_filename)
            continue

        print(f'Processing... {total_files} of {len(os.listdir(path_to_data))}', end="\r", flush=True)
        print('file: ', h5_filename)

        with h5py.File(os.path.join(path_to_data, h5_filename), 'r') as f:
            if total_files == 0:
                print('Keys in the file:', list(f.keys()))

            events_per_file_validation = len(f['cvnmap'][:])

            # Loop over each dataset and append the data
            for key in datasets:
                if key in f:
                    datasets[key].append(f[key][:])

            total_events += events_per_file_validation
            print('events in file: ', events_per_file_validation)
            total_files += 1

    print('total events: ', total_events)

    print('Files read successfully.')
    print('Loaded {} files, and {} total events.'.format(total_files, total_events), flush=True)
    return datasets, total_events, total_files


class IOManager:
    def __init__(self, filename_stub):
        self.filename_stub = filename_stub

    @staticmethod
    def get_det_horn_and_flux_from_string(nova_string):
        """
        :param nova_string (this could be a directory name, file name, etc)
        :type str
        :return: detector.upper(), horn.upper(), and flux.capitalize()
        :rtype: str
        """
        det, horn, flux = '', '', ''
        print("Determining the 'det', 'horn', 'flux' with string........:", nova_string)
        if 'FD' in nova_string:
            print('I found `FD`.')
            det = 'FD'
        elif 'ND' in nova_string:
            print('I found `ND`.')
            det = 'ND'
        else:
            print('ERROR. I did not find a detector exiting......')
            exit()
        print('DETECTOR: {}'.format(det))

        # Get horn...
        print('Determining the horn...')
        if 'FHC' in nova_string:
            print('I found `FHC`.')
            horn = 'FHC'
        elif 'RHC' in nova_string:
            print('I found `RHC`.')
            horn = 'RHC'
        else:
            print('ERROR. I did not find a horn, exiting......')
            exit()
        print('horn: {}'.format(horn))

        # Get flux...
        print('Determining the flux...')
        if 'Fluxswap' in nova_string:
            print('I found `Fluxswap`.')
            flux = 'Fluxswap'
        elif 'Nonswap' in nova_string:
            print('I found `Nonswap`.')
            flux = 'Nonswap'
        else:
            print('ERROR. I did not find a flux, exiting......')
            exit()
        print('flux: {}'.format(flux))

        return det, horn, flux
