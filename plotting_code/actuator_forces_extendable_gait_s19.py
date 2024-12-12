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

# Load the subject 2 .sto file
motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject19\unperturbed.sto"

grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject19\unperturbed_grfs.sto"

contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject19\ankle_perturb_sim_subject19_ForceReporter_forces.sto"

# Load the subject 1 .sto file
# motion_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject0\unperturbedcorrect.sto"

# grf_path_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\unperturbed_grfs_correct.sto"

# contact_time_file = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\force_reporter\Force_report_Heel_r_lateral_midfoot_r_medial_toe_r_correct.sto"

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

print('motion_data', motion_data.shape)

# %% Data manipulation
tibiant_force = motion_data['/forceset/tibant_r'].clip(upper=0.3)
soleus_force = motion_data['/forceset/soleus_r']/4
actuator_force = (tibiant_force + soleus_force)*200
# tibiant_force = motion_data['/forceset/tibant_r']
# soleus_force = motion_data['/forceset/soleus_r']
actuator_force = (tibiant_force + soleus_force)*200
# Select specific columns
heel_r_force = contact_data['contactHeel_r.Sphere.force.Y']
midfoot_r_force = contact_data['contactLateralMidfoot_r.Sphere.force.Y']
toes_r_force = contact_data['contactMedialToe_r.Sphere.force.Y']

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
    'ankle_angle': motion_data['/jointset/ankle_r/ankle_angle_r/value'],
    'ground_force_x': grf_data['ground_force_r_vx'],
    'ground_force_y': grf_data['ground_force_r_vy'],
    'ground_force_z': grf_data['ground_force_r_vz'],
    # 'soleus_force': motion_data['/forceset/soleus_r'],
    'soleus_force': soleus_force,
    'simulated_act_force': actuator_force,
    # 'simulated_act_force': motion_data['/forceset/tibant_r'], 
    'heel_force':heel_r_force_desc,
    'lateral_force': midfoot_r_force_desc,
    'toe_force': toes_r_force_desc
})

# %% Stride length manipulation
def augment_actuator_pattern(dataframe, num_steps=5, stride_variation=0.15, sampling_rate=0.005):
    """
    Augment actuator force pattern with fixed sampling rate and varied stride lengths.
    
    Parameters:
    -----------
    base_force : array-like
        Original force values for a single step
    base_time : array-like
        Original time values for a single step
    num_steps : int
        Number of steps to generate
    stride_variation : float
        Fraction of variation in stride length (0.15 means ±15%)
    sampling_rate : float
        Fixed sampling rate in seconds
    """
    base_time = np.array(dataframe['time'])
    base_force = np.array(dataframe['simulated_act_force'])
    ankle_angle = np.array(dataframe['ankle_angle'])
    grfx = np.array(dataframe['ground_force_x'])
    grfy = np.array(dataframe['ground_force_y'])
    grfz = np.array(dataframe['ground_force_z'])
    heelcontace = np.array(dataframe['heel_force'])
    lateralcontact = np.array(dataframe['lateral_force'])
    toecontact = np.array(dataframe['toe_force'])
    
    # Get original step length
    step_period = base_time[-1] - base_time[0]
    print(f"Original step period: {step_period:.3f}s")
    
    # Initialize a dictionary to store augmented data
    augmented_time = []
    augmented_force = []
    augmented_ankle_angle = []
    augmented_grfx = []
    augmented_grfy = []
    augmented_grfz = []
    augmented_heelcontact = []
    augmented_lateralcontact = []
    augmented_toecontact = []

    current_time = base_time[0]  # Start from the initial time
    
    # Store stride lengths for debugging
    stride_lengths = []
    
    for step in range(num_steps):
        # Generate random stride variation
        stride_factor = 1 + np.random.uniform(-3*stride_variation, 3*stride_variation)
        new_step_period = step_period * stride_factor
        stride_lengths.append(new_step_period)
        
        print(f"Step {step + 1}: stride factor = {stride_factor:.3f}, new period = {new_step_period:.3f}s")
        
        # Calculate number of samples for this step
        num_samples = int(new_step_period / sampling_rate)
        step_time = np.arange(num_samples) * sampling_rate + current_time
        
        # Normalize time arrays for interpolation
        normalized_time = (step_time - current_time) / new_step_period
        original_normalized_time = (base_time - base_time[0]) / step_period
        
        # Interpolate forces
        step_force = np.interp(normalized_time, original_normalized_time, base_force)
        step_ankle_angle = np.interp(normalized_time, original_normalized_time, ankle_angle)
        step_grfx = np.interp(normalized_time, original_normalized_time, grfx)
        step_grfy = np.interp(normalized_time, original_normalized_time, grfy)
        step_grfz = np.interp(normalized_time, original_normalized_time, grfz)
        step_heelcontact = np.interp(normalized_time, original_normalized_time, heelcontace)
        step_lateralcontact = np.interp(normalized_time, original_normalized_time, lateralcontact)
        step_toecontact = np.interp(normalized_time, original_normalized_time, toecontact)

        # Append to result arrays
        augmented_time.extend(step_time)
        augmented_force.extend(step_force)
        augmented_ankle_angle.extend(step_ankle_angle)
        augmented_grfx.extend(step_grfx)
        augmented_grfy.extend(step_grfy)
        augmented_grfz.extend(step_grfz)
        augmented_heelcontact.extend(step_heelcontact)
        augmented_lateralcontact.extend(step_lateralcontact)
        augmented_toecontact.extend(step_toecontact)
        
        # Update current time
        current_time += new_step_period
    
    augmented_data = {
        'time': augmented_time,
        'simulated_act_force': augmented_force,
        'ankle_angle': augmented_ankle_angle,
        'ground_force_x': augmented_grfx,
        'ground_force_y': augmented_grfy,
        'ground_force_z': augmented_grfz,
        'heel_force': augmented_heelcontact,
        'lateral_force': augmented_lateralcontact,
        'toe_force': augmented_toecontact
    }

    print("\nStride length variations:")
    for i, length in enumerate(stride_lengths):
        print(f"Step {i + 1}: {length:.3f}s")
    
    return augmented_data

# Augment actuator force pattern
augmented_data = augment_actuator_pattern(result, num_steps=5, stride_variation=0.15, sampling_rate=0.005)

# %% Plotting actuator forces
plt.figure(figsize=(10, 6))
plt.plot(augmented_data['time'], augmented_data['simulated_act_force'], label='Actuator Force')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')  # Changed to reflect force units
plt.legend()
plt.title('Actuator Force over Time')
plt.grid(True)
plt.show()
# %% Plotting augmented foot timings
plt.figure(figsize=(10, 6))
plt.plot(augmented_data['time'], augmented_data['heel_force'], label='heel contact')
plt.plot(augmented_data['time'], augmented_data['lateral_force'], label='lateral contact')
plt.plot(augmented_data['time'], augmented_data['toe_force'], label='toe contact')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')  # Changed to reflect force units
plt.legend()
plt.title('Actuator Force over Time')
plt.grid(True)
plt.show()

# %% Plotting
plt.figure(figsize=(10, 6))
plt.plot(result['time'], tibiant_force, label='Actuator Force')
plt.plot(result['time'], soleus_force, label='Soleus Force')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')  # Changed to reflect force units
plt.legend()
plt.title('Actuator Force over Time')
plt.grid(True)
plt.show()


# %% Save the augmented data

# Convert dictionary to DataFrame
augmented_data = pd.DataFrame(augmented_data)
# Save the DataFrame to a CSV file
file_path = r"C:\Users\sachi\predsim_tutorial_model\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject19\results_augmented_s19.csv"
augmented_data.to_csv(file_path, index=False)

