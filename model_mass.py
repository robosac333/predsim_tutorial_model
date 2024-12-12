import opensim as osim

# Load the model
model = osim.Model('C:/Users/sachi/predsim_tutorial_model/OpenSimModel/Hamner_modified_stifferContacts/Model/Hamner_modified_stifferContacts_scaled.osim')

# Initialize the model and create a default state
state = model.initSystem()

# Get the total mass of the model
total_mass = model.getTotalMass(state)
print(f'Total mass of the model: {total_mass} kg')