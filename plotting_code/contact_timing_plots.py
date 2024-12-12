# import numpy as np
# import matplotlib.pyplot as plt  
# import os
# import sys
# import pandas as pd

# # Add the main folder to the Python path
# main_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # Go up one directory
# sys.path.append(main_folder_path)


# from utilities import storage2numpy

# # Load the .sto file
# path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\Results\Case_0\Contact_forces\Right Legs1s2s6\Contact_forces_right_leg.sto"
# # path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r"
# data, column_names = storage2numpy(path_file)

# #Convert numpy array to pandas dataframe and add column names
# data = pd.DataFrame(data[1:])

# threshold = 10
# # Extract the heel forces and time
# heel_r_force = data.iloc[:, 2].values
# #thresholding the force values to be 0 or 1
# heel_r_force_desc = np.where(heel_r_force > threshold, 1, 0)
# # Extract the toe forces and time
# toes_r_force = data.iloc[:, 3].values
# # thresholding the force values to be 0 or 1
# toes_r_force_desc = np.where(toes_r_force > threshold, 1, 0)
# # Extract the midfoot forces and time
# midfoot_r_force = data.iloc[:, 4].values
# # thresholding the force values to be 0 or 1
# midfoot_r_force_desc = np.where(midfoot_r_force > threshold, 1, 0)
# time = data.iloc[:, 1].values

# max_time = time[-1]
# normalized_time = time/max_time

# print('heel_r_force', heel_r_force)
# # Plot the forces over time into a single plot
# plt.figure()
# # plt.plot(time, heel_r_force, label='Heel')
# # plt.plot(time, toes_r_force, label='Toes')
# # plt.plot(time, midfoot_r_force, label='Midfoot')
# plt.plot(time, heel_r_force_desc, label='Heel')
# plt.plot(time, toes_r_force_desc, label='Toes')
# plt.plot(time, midfoot_r_force_desc, label='Midfoot')
# plt.xlabel('Time [s]')
# plt.ylabel('Force [N]')
# plt.legend()
# plt.title('Right leg contact forces')
# plt.show()


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
print('sys.path', sys.path)

import utilities 

# Load the .sto file
# motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed.sto"

# grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed_grf.sto"

# contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r.sto"

## For very weaker glutes
motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\force_reporter\Hamner_modified_scaled_ForceReporter_forces.sto"

grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\load_motion_data\GRF.mot"

contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\Contact_forces\Right Legs1s2s6\Contact_forces_right_leg.sto"

ankle_angle_path = r"C:\Users\sachi\predsim_tutorial_model\Results\Case_6\Kinematics\Hamner_modified_scaled_Kinematics_q.sto"
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

ankle_angle_data, ankle_angle_column_names = dataframe_from_sto_file(ankle_angle_path)

print('motion_data', motion_data.shape, "grf_data", grf_data.shape, "contact_data", contact_data.shape)

# %% Data manipulation
tibiant_force = motion_data['tib_ant_r'].clip(upper=0.3)
soleus_force = motion_data['soleus_r']/4
actuator_force = (tibiant_force + soleus_force)*200
# tibiant_force = motion_data['/forceset/tibant_r']
# soleus_force = motion_data['/forceset/soleus_r']
actuator_force = (tibiant_force + soleus_force)*200
# Select specific columns
heel_r_force = contact_data['SmoothSphereHalfSpaceForce_s1_r.Sphere.force.Y']
midfoot_r_force = contact_data['SmoothSphereHalfSpaceForce_s2_r.Sphere.force.Y']
toes_r_force = contact_data['SmoothSphereHalfSpaceForce_s6_r.Sphere.force.Y']

threshold = 10
#thresholding the force values to be 0 or 1
heel_r_force_desc = np.where(heel_r_force > threshold, 1, 0)
# thresholding the force values to be 0 or 1
toes_r_force_desc = np.where(toes_r_force > threshold, 1, 0)
# thresholding the force values to be 0 or 1
midfoot_r_force_desc = np.where(midfoot_r_force > threshold, 1, 0)

max_time = motion_data['time'][len(motion_data['time'])-1]
normalized_time = motion_data['time']/max_time

result = pd.DataFrame({
    'time': motion_data['time'],
    'ankle_angle': ankle_angle_data['ankle_angle_r'],
    'ground_force_x': grf_data['r_ground_force_vx'],
    'ground_force_y': grf_data['r_ground_force_vy'],
    'ground_force_z': grf_data['r_ground_force_vz'],
    # 'soleus_force': motion_data['/forceset/soleus_r'],
    'soleus_force': soleus_force,
    'simulated_act_force': actuator_force,
    # 'simulated_act_force': motion_data['/forceset/tibant_r'], 
    'heel_force': heel_r_force_desc,
    'lateral_force': midfoot_r_force_desc,
    'toe_force': toes_r_force_desc
})

# %% Plot the forces over time into a single plot

plt.figure()
plt.plot(result['time'], motion_data['tib_ant_r'], label='Tibialis Anterior')
# plt.plot(result['time'], motion_data['soleus_r'], label='Soleus')

plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Right leg muscle forces')
plt.show()

# %%
