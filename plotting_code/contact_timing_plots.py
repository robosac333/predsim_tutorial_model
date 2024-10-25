import numpy as np
import matplotlib.pyplot as plt  
import os
import sys
import pandas as pd

# Add the main folder to the Python path
main_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # Go up one directory
sys.path.append(main_folder_path)


from utilities import storage2numpy

# Load the .sto file
path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\Results\Case_0\Contact_forces\Right Legs1s2s6\Contact_forces_right_leg.sto"
data, column_names = storage2numpy(path_file)

#Convert numpy array to pandas dataframe and add column names
data = pd.DataFrame(data[1:])

threshold = 10
# Extract the heel forces and time
heel_r_force = data.iloc[:, 2].values
#thresholding the force values to be 0 or 1
heel_r_force_desc = np.where(heel_r_force > threshold, 1, 0)
# Extract the toe forces and time
toes_r_force = data.iloc[:, 3].values
# thresholding the force values to be 0 or 1
toes_r_force_desc = np.where(toes_r_force > threshold, 1, 0)
# Extract the midfoot forces and time
midfoot_r_force = data.iloc[:, 4].values
# thresholding the force values to be 0 or 1
midfoot_r_force_desc = np.where(midfoot_r_force > threshold, 1, 0)
time = data.iloc[:, 1].values

max_time = time[-1]
normalized_time = time/max_time

print('heel_r_force', heel_r_force)
# Plot the forces over time into a single plot
plt.figure()
# plt.plot(time, heel_r_force, label='Heel')
# plt.plot(time, toes_r_force, label='Toes')
# plt.plot(time, midfoot_r_force, label='Midfoot')
plt.plot(time, heel_r_force_desc, label='Heel')
plt.plot(time, toes_r_force_desc, label='Toes')
plt.plot(time, midfoot_r_force_desc, label='Midfoot')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Right leg contact forces')
plt.show()


