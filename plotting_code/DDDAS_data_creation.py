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
# motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\load_motion_data\motion_extend.mot"

# grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\load_motion_data\GRF_extend.mot"

# contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\force_reporter\Hamner_modified_scaled_ForceReporter_forces_extended.sto"

# motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed.sto"

# grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed_grf.sto"

# contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r.sto"

motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbedcorrect.sto"

grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed_grfs_correct.sto"

contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r_correct.sto"

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
    'ground_force_x': grf_data['ground_force_r_vx'],
    'ground_force_y': grf_data['ground_force_r_vy'],
    'ground_force_z': grf_data['ground_force_r_vz'],
    'simulated_act_force': grf_data['ground_force_r_vy']//10,
    'heel_force': contact_data['contactHeel_r.Sphere.force.Y'],
    'lateral_force': contact_data['contactLateralMidfoot_r.Sphere.force.Y'],
    'toe_force': contact_data['contactMedialToe_r.Sphere.force.Y']
})

# print('result', result.shape)

# result = pd.DataFrame({
#     'time': motion_data['time'],
#     'ankle_angle': motion_data['ankle_angle_r'],
#     'ground_force_x': grf_data['r_ground_force_vx'],
#     'ground_force_y': grf_data['r_ground_force_vy'],
#     'ground_force_z': grf_data['r_ground_force_vz'],
#     'simulated_act_force': grf_data['r_ground_force_vy']//10,
#     'heel_force': contact_data['SmoothSphereHalfSpaceForce_s1_r.Sphere.force.Y'],
#     'lateral_force': contact_data['SmoothSphereHalfSpaceForce_s2_r.Sphere.force.Y'],
#     'toe_force': contact_data['SmoothSphereHalfSpaceForce_s5_r.Sphere.force.Y']
# })

print('result', result.shape)


# %% Plot the data

plt.figure()
plt.plot(result['time'], result['ankle_angle'], label='Ankle angle')
plt.xlabel('Time [s]')
plt.ylabel('Ankle angle [rad]')
plt.legend()
plt.title('Ankle angle over time')
plt.show()

# plt.figure()
# plt.plot(result['time'], result['ground_force'], label='Ground force')
# plt.xlabel('Time [s]')
# plt.ylabel('Ground force [N]')
# plt.legend()
# plt.title('Ground force over time')
# plt.show()

# data = pd.DataFrame(contact_data.iloc[:, 1:].values, columns=result.columns[2:])
data = result

threshold = 10
# Extract the heel forces and time
heel_r_force = data.iloc[:, 6].values
#thresholding the force values to be 0 or 1
heel_r_force_desc = np.where(heel_r_force > threshold, 1, 0)
# Extract the toe forces and time
toes_r_force = data.iloc[:, 7].values
# thresholding the force values to be 0 or 1
toes_r_force_desc = np.where(toes_r_force > threshold, 1, 0)
# Extract the midfoot forces and time
midfoot_r_force = data.iloc[:, 8].values
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
    'ground_force_x': result['ground_force_x'],
    'ground_force_y': result['ground_force_y'],
    'ground_force_z': result['ground_force_z'],
    'simulated_act_force': result['simulated_act_force'],
    'heel_strike': heel_r_force_desc,
    'midfoot_strike': toes_r_force_desc,
    'toe_strike': midfoot_r_force_desc
})

file_path = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\result.csv"
final_data.to_csv(file_path, index=False)

# %% Data Augmentation by opening the file and adding the data
file_path = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\result.csv"

import pandas as pd
import numpy as np

final_data = pd.read_csv(file_path)

def extend_timeseries(df, num_repetitions, time_intervals):
    """
    Extend a timeseries dataframe by repeating it with different time intervals.
    
    Parameters:
    df (pandas.DataFrame): Original dataframe with 'time' column
    num_repetitions (int): Number of times to repeat the data
    time_intervals (list): List of time intervals for each repetition
    
    Returns:
    pandas.DataFrame: Extended dataframe with modified time columns
    """
    if len(time_intervals) != num_repetitions:
        raise ValueError("Number of time intervals must match number of repetitions")
        
    # Get the original time step
    original_timestep = df['time'].iloc[1] - df['time'].iloc[0]
    
    # Create empty list to store dataframes
    dfs = []
    
    # Process each repetition
    for rep_idx, interval in enumerate(time_intervals):
        # Create a copy of the original dataframe
        temp_df = df.copy()
        
        # Calculate new time values
        num_points = len(temp_df)
        new_times = np.arange(0, num_points * interval, interval)
        
        # Update time column
        temp_df['time'] = new_times + (rep_idx * new_times[-1] + interval)
        
        # Add repetition identifier if needed
        temp_df['repetition'] = rep_idx
        
        dfs.append(temp_df)
    
    # Concatenate all dataframes
    extended_df = pd.concat(dfs, ignore_index=True)
    
    return extended_df

# Define number of repetitions and intervals
num_repetitions = 4
time_intervals = [0.007, 0.009, 0.011, 0.011]  # Different intervals for each repetition

# Extend the dataset
# loaded_data = extend_timeseries(final_data, num_repetitions, time_intervals)

# %% Load the data
import matplotlib.pyplot as plt

file_path 
loaded_data = pd.read_csv(file_path)

loaded_data = extend_timeseries(loaded_data, num_repetitions, time_intervals)

# Plot the loaded data
plt.figure()
plt.plot(loaded_data['time'], loaded_data['ankle_angle'], label='Ankle angle')
plt.xlabel('Time [s]')
plt.ylabel('Ankle angle [rad]')
plt.legend()
plt.title('Ankle angle over time')
plt.show()

plt.figure()
plt.plot(loaded_data['time'], loaded_data['simulated_act_force'], label='Actuator Force')
plt.xlabel('Time [s]')
plt.ylabel('Simiulated Actuator Force [N]')
plt.legend()
plt.title('Actuator Force over time')
plt.show()

plt.figure()
plt.plot(loaded_data['time'], loaded_data['ground_force_x'], label='Ground Force X')
plt.plot(loaded_data['time'], loaded_data['ground_force_y'], label='Ground Force Y')
plt.plot(loaded_data['time'], loaded_data['ground_force_z'], label='Ground Force Z')
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



# %% Save the data

file_path = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\results_extended.csv"

loaded_data.to_csv(file_path, index=False)