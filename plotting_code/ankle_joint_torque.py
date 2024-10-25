
# %% Import packages.
import os
import numpy as np
import matplotlib.pyplot as plt  

# %% User inputs
cases = ['0']

# %% Paths
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
# %% Joint positions.
joints = optimaltrajectories[cases[0]]['joints']
# jointToPlot = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 
#                'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 
#                'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
#                'knee_angle_r',  'ankle_angle_r', 
#                'subtalar_angle_r', 'mtp_angle_r', 
#                'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
#                'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r']
jointToPlot = ['ankle_angle_r']
from utilities import getJointIndices
idxJointsToPlot = getJointIndices(joints, jointToPlot)
NJointsToPlot = len(jointToPlot) 

# Create the directory if it doesn't exist
save_path = r'C:\Users\sjd3333\Desktop\predsim_tutorial\OpenSimModel\Very_weak_glutes\Model\plots'
if not os.path.exists(save_path):
    os.makedirs(save_path)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Ankle Angle (Right) Joint')

color = iter(plt.cm.rainbow(np.linspace(0, 1, len(cases))))
plotExperimental = True

for case in cases:
    c_joints = optimaltrajectories[case]['joints']
    if 'ankle_angle_r' not in c_joints:
        continue
    
    c_joint_idx = c_joints.index('ankle_angle_r')
    ax.plot(optimaltrajectories[case]['GC_percent'],
            optimaltrajectories[case]['coordinate_values'][c_joint_idx:c_joint_idx+1, :].T, 
            c=next(color), label='case_' + case)

# if plotExperimental:
#     ax.fill_between(experimentalData[model_name]["kinematics"]["positions"]["GC_percent"],
#                     experimentalData[model_name]["kinematics"]["positions"]["mean"]["ankle_angle_r"] + 2*experimentalData[model_name]["kinematics"]["positions"]["std"]["ankle_angle_r"],
#                     experimentalData[model_name]["kinematics"]["positions"]["mean"]["ankle_angle_r"] - 2*experimentalData[model_name]["kinematics"]["positions"]["std"]["ankle_angle_r"],
#                     facecolor='grey', alpha=0.4, label='Experimental Data (±2 SD)')

ax.set_xlabel('Gait cycle (%)')
ax.set_ylabel('Angle (deg)')
ax.set_title('Ankle Angle (Right)')
ax.legend(loc='upper right')

fig.tight_layout()  # Use tight layout to prevent label clipping
fig.savefig(os.path.join(save_path, 'ankle_angle_r.png'), dpi=300)

# plt.tight_layout()
# plt.show()

# %% Joint torques.
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Ankle Angle (Right) Joint Toruqe')

color = iter(plt.cm.rainbow(np.linspace(0, 1, len(cases))))
plotExperimental = True

for case in cases:
    c_joints = optimaltrajectories[case]['joints']
    if 'ankle_angle_r' not in c_joints:
        continue
    
    c_joint_idx = c_joints.index('ankle_angle_r')
    ax.plot(optimaltrajectories[case]['GC_percent'],
            optimaltrajectories[case]['joint_torques'][c_joint_idx:c_joint_idx+1, :].T, 
            c=next(color), label='case_' + case)
    


# if plotExperimental:
#     ax.fill_between(experimentalData[model_name]["kinetics"]["GC_percent"],
#                     experimentalData[model_name]["kinetics"]["mean"]["ankle_angle_r"] + 2*experimentalData[model_name]["kinetics"]["std"]["ankle_angle_r"],
#                     experimentalData[model_name]["kinetics"]["mean"]["ankle_angle_r"] - 2*experimentalData[model_name]["kinetics"]["std"]["ankle_angle_r"],
#                     facecolor='grey', alpha=0.4, label='Experimental Data (±2 SD)')

ax.set_xlabel('Gait cycle (%)')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Ankle Angle (Right) Torque')
# ax.legend(loc='upper right')

fig.tight_layout()  # Use tight layout to prevent label clipping
fig.savefig(os.path.join(save_path, 'ankle_angle_r_torque.png'), dpi=300)
# plt.tight_layout()
# plt.show()
