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
motion_path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed.sto"

grf_path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed_grf.sto"

contact_time_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r.sto"

def dataframe_from_sto_file(path_file):
    data, column_names = utilities.storage2numpy(path_file)
    new_array = np.zeros((data.shape[0], len(column_names)))
    for i, data in enumerate(data):
        row = list(data)
        new_array[i] = row
    
    pandas_data = pd.DataFrame(new_array, columns=column_names)
    return pandas_data, column_names

motion_data, motion_column_names = dataframe_from_sto_file(motion_path_file)
grf_data, grf_column_names = dataframe_from_sto_file(grf_path_file)
contact_data, contact_column_names = dataframe_from_sto_file(contact_time_file)

print('motion_data', motion_data.shape, 'grf_data', grf_data.shape, 'contact_data', contact_data.shape)
        

# %% Data manipulation

# Select specific columns
result = pd.DataFrame({
    'time': motion_data['time'],
    'ankle_angle': motion_data['/jointset/ankle_r/ankle_angle_r/value'],
    'ground_force': grf_data['ground_force_r_vy']//10,
    'heel_force': contact_data['contactHeel_r.Sphere.force.Y'],
    'lateral_force': contact_data['contactLateralMidfoot_r.Sphere.force.Y'],
    'toe_force': contact_data['contactMedialToe_r.Sphere.force.Y']
})

print('result', result.shape)


# %% Plot the data

plt.figure()
plt.plot(result['time'], result['ankle_angle'], label='Ankle angle')
plt.xlabel('Time [s]')
plt.ylabel('Ankle angle [rad]')
plt.legend()
plt.title('Ankle angle over time')
plt.show()

plt.figure()
plt.plot(result['time'], result['ground_force'], label='Ground force')
plt.xlabel('Time [s]')
plt.ylabel('Ground force [N]')
plt.legend()
plt.title('Ground force over time')
plt.show()

data = pd.DataFrame(contact_data.iloc[:, 1:].values, columns=result.columns[2:])

threshold = 10
# Extract the heel forces and time
heel_r_force = data.iloc[:, 1].values
#thresholding the force values to be 0 or 1
heel_r_force_desc = np.where(heel_r_force > threshold, 1, 0)
# Extract the toe forces and time
toes_r_force = data.iloc[:, 2].values
# thresholding the force values to be 0 or 1
toes_r_force_desc = np.where(toes_r_force > threshold, 1, 0)
# Extract the midfoot forces and time
midfoot_r_force = data.iloc[:, 3].values
# thresholding the force values to be 0 or 1
midfoot_r_force_desc = np.where(midfoot_r_force > threshold, 1, 0)
time = data.iloc[:, 0].values

max_time = time[-1]
normalized_time = time/max_time
plt.figure()
plt.plot(time, heel_r_force_desc, label='Heel')
plt.plot(time, toes_r_force_desc, label='Toes')
plt.plot(time, midfoot_r_force_desc, label='Midfoot')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Right leg contact forces')
plt.show()

# %% Save the data

final_data = pd.DataFrame({
    'time': result['time'],
    'ankle_angle': result['ankle_angle'],
    'ground_force': result['ground_force'],
    'heel_strike': heel_r_force_desc,
    'midfoot_strike': toes_r_force_desc,
    'toe_strike': midfoot_r_force_desc
})

file_path = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\result.csv"
final_data.to_csv(file_path, index=False)

# %% Load the data

loaded_data = pd.read_csv(file_path)

# Plot the loaded data
plt.figure()
plt.plot(loaded_data['time'], loaded_data['ankle_angle'], label='Ankle angle')
plt.xlabel('Time [s]')
plt.ylabel('Ankle angle [rad]')
plt.legend()
plt.title('Ankle angle over time')
plt.show()

plt.figure()
plt.plot(loaded_data['time'], loaded_data['ground_force'], label='Ground force')
plt.xlabel('Time [s]')
plt.ylabel('Ground force [N]')
plt.legend()
plt.title('Ground force over time')
plt.show()


plt.figure()
plt.plot(loaded_data['time'], loaded_data['heel_strike'], label='Heel')
plt.plot(loaded_data['time'], loaded_data['midfoot_strike'], label='Midfoot')
plt.plot(loaded_data['time'], loaded_data['toe_strike'], label='Toe')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Right leg contact forces')
plt.show()

