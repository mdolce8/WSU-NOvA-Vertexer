# data_processing.py
# Useful utilities for doing the training.


from numpy import ndarray, array

# coordinate conversion functions. These are global.
# Far Detector conversions.
def convert_x_pixelmap_to_fd_vtx_x(x_pixelmap_array, firstcellx_array):
  return (x_pixelmap_array + firstcellx_array - 192) * 3.97

def convert_y_pixelmap_to_fd_vtx_y(y_pixelmap_array, firstcelly_array):
  return (y_pixelmap_array + firstcelly_array - 191) * 3.97

def convert_z_pixelmap_to_fd_vtx_z(z_pixelmap_array, firstplane_array):
  return (z_pixelmap_array + firstplane_array) * 6.664

def convert_fd_vtx_x_to_pixelmap(vtx_x_array, firstcellx_array):
    """
    :param vtx_x_array: `vtx.x` -- x in detector coordinates.
    :param firstcellx_array: `firstcellx` -- first x cell in pixelmap coordinates
    :return: x pixelmap coordinate
    """
    print('Converting x coordinate...')
    return vtx_x_array / 3.97 - firstcellx_array + 192
def convert_fd_vtx_y_to_pixelmap(vtx_y_array, firstcelly_array):
    """
    :param vtx_y_array: `vtx.y` -- y in detector coordinates.
    :param firstcelly_array: `firstcelly` -- first y cell in pixelmap coordinates
    :return: y pixelmap coordinate
    """
    print('Converting y coordinate...')
    return vtx_y_array / 3.97 - firstcelly_array + 191
def convert_fd_vtx_z_to_pixelmap(vtx_z_array, firstplane_array):
    """
    :param vtx_z_array: `vtx.z` -- z in detector coordinates.
    :param firstplane_array: `firstplane` -- first plane in pixelmap coordinates
    :return: z pixelmap coordinate
    """
    print('Converting z coordinate...')
    return vtx_z_array / 6.664 - firstplane_array

# Near Detector conversions.
def convert_nd_vtx_x_to_pixelmap(vtx_x_array, firstcellx_array):
    return vtx_x_array / 3.99 - firstcellx_array + 48
def convert_nd_vtx_y_to_pixelmap(vtx_y_array, firstcelly_array):
    return vtx_y_array / 3.97 - firstcelly_array + 47
def convert_nd_vtx_z_to_pixelmap(vtx_z_array, firstplane_array):
    return vtx_z_array / 6.61 - firstplane_array

def convert_lists_to_nparray(dsets) -> dict:
    """
    Convert the lists in dsets to numpy arrays
    :param dsets:
    :return:
    """
    print("Converting to lists in 'datasets' to numpy array...")
    dsets = {key: array(dsets[key]) for key in dsets}
    return dsets


def print_input_data(d_tr, d_te, d_va) -> None:
    """
    print out the shapes of the input datasets into model
    :param d_tr: data_train
    :param d_te: data_test
    :param d_va: data_val
    :return: None
    """
    print('Final printout of shape before feeding into network......')
    print('training: (after final reshaping)')
    print("data_train['xz'].shape: ", d_tr['xz'].shape)
    print("data_train['yz'].shape: ", d_tr['yz'].shape)
    if d_va:
        print("data_val['xz'].shape: ", d_va['xz'].shape)
        print("data_val['yz'].shape: ", d_va['yz'].shape)
    print('testing:')
    print("test_train['xz'].shape: ", d_te['xz'].shape)
    print("data_test['xz'].shape: ", d_te['xz'].shape)
    return None


class ConvertFarDetCoords:
    def __init__(self, det, coordinate):
        self.det = det
        assert self.det.lower() == 'fd'
        self.coordinate = coordinate
        assert self.coordinate in {'x', 'y', 'z'}
        self.c_map_to_vtx = {
            'x' : convert_x_pixelmap_to_fd_vtx_x,
            'y' : convert_y_pixelmap_to_fd_vtx_y,
            'z' : convert_z_pixelmap_to_fd_vtx_z,
        }
        self.c_map_to_pixelmap = {
            'x' : convert_fd_vtx_x_to_pixelmap,
            'y' : convert_fd_vtx_y_to_pixelmap,
            'z' : convert_fd_vtx_z_to_pixelmap,
        }

    def convert_pixelmap_to_fd_vtx(self, pixelmap_array, first_hit_array):
        if self.coordinate in self.c_map_to_vtx.keys():
            return self.c_map_to_vtx[self.coordinate](pixelmap_array, first_hit_array)
        else:
            raise ValueError(f'Coordinate {self.coordinate} is not valid.')

    def convert_fd_vtx_to_pixelmap(self, vtx_array, first_hit_array):
        if self.coordinate in self.c_map_to_pixelmap.keys():
            return self.c_map_to_pixelmap[self.coordinate](vtx_array, first_hit_array)
        else:
            raise ValueError(f'Coordinate {self.coordinate} is not valid.')


class Debug:
    def __init__(self, array_file):
        self.arr = array_file

    def printout_type(self) -> None:
        """
        Check the array. There should be 8 files in a given array (1 for each file).
        :return: str, confirming you have numpy.arrays
        """
        print('type of outer array: ', type(self.arr))
        print('shape of outer array: ', self.arr.shape)
        for file in range(0, len(self.arr)):
            if file == 0:
                print('file: ', file)
                print('type of inner array: ', type(self.arr[file]))
                print('shape of inner array: ', self.arr[file].shape)
                assert (type(self.arr[file]) is ndarray), "array must be a numpy array"
                print('-------------------')
        print('All file entries for the array have been checked -- they are np.ndarray')
        return None


class DataCleaning:
    def __init__(self, first_hit_array, coordinate):
        self.first_hit_array = array(first_hit_array)  # Convert to np.array if not already
        self.first_hit_array = first_hit_array
        self.coordinate = coordinate
        assert self.coordinate in {'x', 'y', 'z'}

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
    def remove_unsigned_ints(self, single_file=False) -> array:
        """
        Function to remove unsigned ints.
        Has an option for when using a single file. (e.g. for 'testsize' training
        AND single inference file model prediction cases).
        :param single_file: boolean, default False.
        :return: list of firstcell/firstplane.
        :rtype: list
        """
        # some debugging, to be sure...
        print(f"You selected 'single_file'={single_file}. ")
        print('Addressing the bug in the Prod5.1 h5 trimmed files.\n'
              '1. Adding 40,\n'
              '2. Converting firstcellx, firstcelly, firstplane to `int`,\n'
              '3. Subtracting 40......')
        if not single_file:
            for fileIdx in range(len(self.first_hit_array.shape)):
                print('Begin converting entries in file: ', fileIdx)
                if self.coordinate in {'x', 'y'}:
                    if fileIdx == 0:  # print only first file of work...
                        print('self.first_hit_array[fileIdx].shape', self.first_hit_array[fileIdx].shape)
                        print('self.first_hit_array[fileIdx][100] at start: ', self.first_hit_array[fileIdx][100], type(self.first_hit_array[fileIdx][100]))

                    self.first_hit_array[fileIdx] += 40
                    self.first_hit_array[fileIdx] = array(self.first_hit_array[fileIdx], dtype='int')
                    if fileIdx == 0:
                        print('self.first_hit_array[fileIdx][100] after conversion + 40 addition: ', self.first_hit_array[fileIdx][100], type(self.first_hit_array[fileIdx][100]))

                    self.first_hit_array[fileIdx] -= 40
                    if fileIdx == 0:
                        print('self.first_hit_array[fileIdx][1]] after conversion + 40 subtraction: ', self.first_hit_array[fileIdx][100], type(self.first_hit_array[fileIdx][100]))

                elif self.coordinate == 'z':
                    if fileIdx == 0:
                        print('converting the firstplane to `int`....')
                    self.first_hit_array[fileIdx] = array(self.first_hit_array[fileIdx], dtype='int')  # not strictly necessary, Erin doesn't do it...

                else:
                    ValueError(f'coordinate {self.coordinate} is not valid.')
        else:
            print('Begin converting entries in SINGLE file: ')
            if self.coordinate in {'x', 'y'}:
                print('self.first_hit_array.shape', self.first_hit_array.shape)
                print('self.first_hit_array[100] at start: ', self.first_hit_array[100],
                      type(self.first_hit_array[100]))

                self.first_hit_array += 40
                self.first_hit_array = array(self.first_hit_array, dtype='int')
                print('self.first_hit_array[100] after conversion + 40 addition: ',
                      self.first_hit_array[100], type(self.first_hit_array[100]))

                self.first_hit_array -= 40
                print('self.first_hit_array[1]] after conversion + 40 subtraction: ',
                      self.first_hit_array[100], type(self.first_hit_array[100]))

            elif self.coordinate == 'z':
                print('converting the firstplane to `int`....')
                self.first_hit_array = array(self.first_hit_array,
                                                      dtype='int')  # not strictly necessary, Erin doesn't do it...

            else:
                ValueError(f'coordinate {self.coordinate} is not valid.')
        return self.first_hit_array


    @staticmethod
    # or select_y_centered_cvnmaps() ...?
    def remove_uncentered_y_cvnmaps(vtx_coords_cvnmap, lower_limit=20, upper_limit=50) -> (array, int):
        """
        Function to remove events that have Y vertex that is not in the center.
        Investigated by Abdul, values within [20, 50] are correct cvnmaps.
        User can set the edges they desire themselves with upper and lower_limit.
        These are likely improperly drawn cvnmaps.
        :param vtx_coords_cvnmap: np.array (array of all three vtx locations in cvnmap coords)
        :param lower_limit: int (lower value of position to cut)
        :param upper_limit: int (upper value of position to cut)
        :return: vtx_coords_y_centered, events_removed
        """
        # Remove the events that have un-centered Y cvnmaps
        vtx_y_cvnmap_center_filter = (vtx_coords_cvnmap[:, 1] > lower_limit) & (vtx_coords_cvnmap[:, 1] < upper_limit)
        vtx_coords_y_centered = vtx_coords_cvnmap[vtx_y_cvnmap_center_filter]
        events_removed = len(vtx_coords_cvnmap) - len(vtx_coords_y_centered)
        print('Events removed that are not centered in Y axis on cvnmap: ', events_removed)
        return vtx_coords_y_centered, events_removed