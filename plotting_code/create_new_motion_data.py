# %% Create Data Frame from numpy array and add column names
import numpy as np

import matplotlib.pyplot as plt  
import os
import sys
import pandas as pd


import os
import sys

# Add the main folder to the Python path
current_directory = os.path.dirname(os.path.abspath(__file__))  # Path of the current script
sys.path.append(current_directory)
# print('sys.path', sys.path)

import utilities 

# Load the .sto file
motion_path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed.sto"

grf_path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed_grfs.sto"

motion_data, motion_column_names = utilities.storage2numpy(motion_path_file)

grf_data, grf_column_names = utilities.storage2numpy(grf_path_file)

#Convert numpy array to pandas dataframe and add column names
motion_data = pd.DataFrame(motion_data)

grf_data = pd.DataFrame(grf_data)

# print('motion_data', motion_data.iloc[:, 0])

start_idx = np.where(motion_data.iloc[:, 0] == 4.49)[0][0]

# Create array of incremental values
incremental_values = np.arange(0, len(motion_data) - start_idx) * 0.05 + 4.49

motion_data.loc[start_idx:, 'time'] = incremental_values

motion_data = motion_data.to_numpy()
# print('motion_data', grf_data.iloc[:, 0])

# print('incremental_values', incremental_values)

storage_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin.sto"


# %% From numpy array to storage file.
def numpy2storage(labels, data, storage_file):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    f.write('name %s\n' % storage_file)
    f.write('datacolumns %d\n' % data.shape[1])
    f.write('datarows %d\n' % data.shape[0])
    f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()

numpy2storage(motion_column_names, motion_data, storage_file)
print('motion_data.shape', motion_data.shape[1],'motion_column_names', len(motion_column_names))
