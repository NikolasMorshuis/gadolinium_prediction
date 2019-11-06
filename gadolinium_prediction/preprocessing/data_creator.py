import numpy as np
import util.import_functions as import_functions
import sys
from multiprocessing import Pool
import os
import util.util as ut
import glob

to_load = ut.Dataoperations()
files = to_load.get_subdirs('./data', level=3)
print(files)  # should print a list of


def save_function(path):

    # load the data and define the size of each dimension:
    y = import_functions.preprocessing()
    data, _, _, _, _, _ = y.import_data_nifti3d(path, crop_img=True, difference=True, name_T1='T1', name_T1CE='T1_KM')
    n_channels = np.shape(data[0][0])[0]
    n_dim1 = np.shape(data[0][0])[1]
    n_dim2 = np.shape(data[0][0])[2]
    n_dim3 = np.shape(data[0][0])[3]

    # Transform data to nicely shaped numpy-data: [1, channels, dim1, dim2, dim3]
    tot_matrix = np.zeros((1, n_channels, n_dim1, n_dim2, n_dim3))
    seg_matrix = np.zeros((1, 2, n_dim1, n_dim2, n_dim3))
    data = np.array(data)
    tot_matrix[0, :, :, :, :] = data[0, 0]  # data
    seg_matrix[0, 0, :, :, :] = data[0, 1]  # label
    seg_matrix[0, 1, :, :, :] = data[0, 2]  # mask

    # change the save location if you like
    location = path

    if not os.path.exists(location):
        os.makedirs(location)

    np.save(os.path.join(location, 'data.npy'), tot_matrix)
    np.save(os.path.join(location, 'gt.npy'), seg_matrix)
    print('data and groundtruth saved here: {}'.format(path))

pool = Pool(processes=4)
pool.map(save_function, files)
