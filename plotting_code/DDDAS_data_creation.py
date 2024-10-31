# %% Create Data Frame from numpy array and add column names
import numpy as np

import matplotlib.pyplot as plt  
import os
import sys
import pandas as pd


import os
import sys

# Add the main folder to the Python path
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_directory)
# print('sys.path', sys.path)

import utilities 

# Load the .sto file
motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed.sto"

grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed_grfs.sto"

contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r.sto"

# Load the .sto files in numpy arrays
motion_data, motion_column_names = utilities.storage2numpy(motion_path_file)

grf_data, grf_column_names = utilities.storage2numpy(grf_path_file)

contact_data, column_names = utilities.storage2numpy(contact_time_file)

motion_data = utilities.storage2df(motion_path_file, motion_column_names)

grf_data = utilities.storage2df(grf_path_file, grf_column_names)

contact_data = utilities.storage2df(contact_time_file, column_names)

print('motion_data', motion_data.shape, 'grf_data', grf_data.shape, 'contact_data', contact_data.shape)
#Convert numpy array to pandas dataframe and add column names
# motion_data = pd.DataFrame(motion_data, columns=motion_column_names)

# grf_data = pd.DataFrame(grf_data, columns=grf_column_names)

# contact_data = pd.DataFrame(contact_data, columns=column_names)

# print('motion_data', motion_data.shape, 'grf_data', grf_data.shape, 'contact_data', contact_data.shape)

# print("motion_data", motion_data)
print( motion_data['/jointset/ankle_r/ankle_angle_r/value'])
# %% Data manipulation

# # Select specific columns
# result = pd.DataFrame({
#     'time': motion_data['time'],
#     'ankle_angle': motion_data['/jointset/ankle_r/ankle_angle_r/value'],
#     'ground_force': grf_data['ground_force_r_vy'],
#     'heel_force': contact_data['contactHeel_r.Sphere.force.Y'],
#     'lateral_force': contact_data['contactLateralMidfoot_r.Sphere.force.Y'],
#     'toe_force': contact_data['contactMedialToe_r.Sphere.force.Y']
# })

# print('result', result.shape)

# # %% Write the data to a storage csv file

# result.to_csv(r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\result.csv", index=False)

