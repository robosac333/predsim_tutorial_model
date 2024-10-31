import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

file_path = r"C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Newmodelextendablegait\Model\perturbed_torque0_time60_rise10_fall5\subject01\sachin_unperturbed.sto"
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