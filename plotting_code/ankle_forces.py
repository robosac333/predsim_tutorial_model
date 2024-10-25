import os
import numpy as np
import matplotlib.pyplot as plt  

cases = ['0']

pathMain = os.getcwd()
# Load results
pathTrajectories = os.path.join(pathMain, 'Results')
optimaltrajectories = np.load(os.path.join(pathTrajectories, 
                                           'optimalTrajectories.npy'),
                              allow_pickle=True).item()
# Load experimental data
model_name = 'Hamner_modified'
pathData = os.path.join(pathMain, 'OpenSimModel', model_name)
experimentalData = np.load(os.path.join(pathData, 'experimentalData.npy'),
                           allow_pickle=True).item()

from utilities import getJointIndices
case = '0'
GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
GRFToPlot = ['GRF_x_r', 'GRF_y_r', 'GRF_z_r', 'GRF_x_l','GRF_y_l', 'GRF_z_l']
NGRFToPlot = len(GRFToPlot)
idxGRFToPlot = getJointIndices(GRF_labels, GRFToPlot)
GRF_y_force = optimaltrajectories[case]['GRF'][idxGRFToPlot[1]:idxGRFToPlot[1]+1, :].T

from utilities import storage2numpy

# Load the .sto file
path_file = r"C:\Users\sjd3333\Desktop\predsim_tutorial\Results\Case_0\body_kinematics\Hamner_modified_scaled_BodyKinematics_acc_global.sto"
data = storage2numpy(path_file)

calcn_r_accln_from_ground = data['calcn_r_Oy']

# print(calcn_r_accln_from_ground)

print(GRF_y_force)

import numpy as np
# foot_mass = calcn_r + talus_r + toes_r masses
foot_mass = 1.03 + 0.08 + 0.17
g = 9.81
# Ensure GRF_y_force is a 1D array for easy arithmetic with Foot_force
GRF_y_force = np.squeeze(GRF_y_force)  # This removes extra dimensions

# Make sure calcn_r_accln_from_ground and GRF_y_force have the same length
assert len(calcn_r_accln_from_ground) == len(GRF_y_force), "Mismatched lengths"

# Recalculate Foot force
Foot_force = foot_mass * calcn_r_accln_from_ground

if len(GRF_y_force.shape) > 1:
    GRF_y_force = GRF_y_force[:, 0]  

# Foot weight
foot_weight = np.full((len(Foot_force),), foot_mass * g)

# Ankle force calculation
Force_on_ankle = -Foot_force + GRF_y_force - foot_weight

# Ensure time has the same shape as forces
time = optimaltrajectories[case]['time'].T
assert len(time) == len(Force_on_ankle)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Ankle Force (Right) Joint')

# Plot ankle force vs time
ax.plot(time, Force_on_ankle, label=f'case_{case}')

# Labels and formatting
ax.set_xlabel('Time (s)')
ax.set_ylabel('Force (N)')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Time shape: {time.shape}")
print(f"GRF_y_force shape: {GRF_y_force.shape}")
print(f"Foot_force shape: {Foot_force.shape}")
print(f"Force_on_ankle shape: {Force_on_ankle.shape}")


