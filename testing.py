from change_gait_params import calculate_start_conditions, calculate_end_conditions, calculate_coeffs_list
import numpy as np
import matplotlib.pyplot as plt
from predict_energy import load_and_preprocess_data_single
import pandas as pd

data_1803_delta_40 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_40.csv", 1)
data_1803_delta_30 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_30.csv", 1)
data_1803_delta_20 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_20.csv", 1)
combined_data = pd.concat([data_1803_delta_40, data_1803_delta_30, data_1803_delta_20], ignore_index=True)

"""
alpha_h_current = 30 * np.pi / 180
omega_h_current = 150 * np.pi / 180
delta_h_current = 40 * np.pi / 180
n = 10

alpha_h_target = 10 * np.pi / 180
omega_h_target = 210 * np.pi / 180
delta_h_target = 40 * np.pi / 180

t = 0
T = 1

start_conditions = calculate_start_conditions(alpha_h_current, omega_h_current, delta_h_current, n, t)
end_conditions = calculate_end_conditions(alpha_h_target, omega_h_target, delta_h_target, n, t, T)
coeffs_list = calculate_coeffs_list(start_conditions, end_conditions, T)

time_steps = []
phi_refs = []
# Increment t in the loop
dt = 0.1  # Time step increment
while t < 20:
    elapsed_time = t
    for i in range(n):
        alpha_h_current = alpha_h_target
        omega_h_current = omega_h_target
        delta_h_current = delta_h_target
        phi_ref = alpha_h_current * np.sin(omega_h_current * t + i * delta_h_current)
            # Assuming you want to plot the last phi_ref of each time step
    
    # Append current time and phi_ref to their respective lists
    time_steps.append(t)
    phi_refs.append(phi_ref)
    
    t += dt  # Increment time


# Convert phi_refs to a numpy array for element-wise operations
phi_refs_array = np.array(phi_refs)
# Now you can apply the conversion to degrees directly on the array
phi_refs_degrees = phi_refs_array * 180 / np.pi

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_steps, phi_refs_degrees, label="phi_ref over time (Degrees)")
plt.xlabel("Time")
plt.ylabel("phi_ref (Degrees)")
plt.title("Transition of Gait Parameters")
plt.legend()
plt.grid(True)
plt.show()


# Assuming combined_data is your pandas DataFrame from the concatenated datasets
max_avg_velocity_idx = combined_data['average_velocity'].idxmax()  # Find the index of the max average_velocity
max_avg_velocity_entry = combined_data.loc[max_avg_velocity_idx]  # Retrieve the entry at that index
combined_data = combined_data[combined_data['average_velocity'] < 1]

combined_data['normalized_velocity'] = (combined_data['average_velocity'] - combined_data['average_velocity'].min()) / (combined_data['average_velocity'].max() - combined_data['average_velocity'].min())
combined_data['normalized_energy'] = (combined_data['average_energy'] - combined_data['average_energy'].min()) / (combined_data['average_energy'].max() - combined_data['average_energy'].min())

combined_data['velocity_energy_ratio'] = combined_data['normalized_velocity'] - combined_data['normalized_energy']

sorted_data = combined_data.sort_values(by='velocity_energy_ratio', ascending=False)

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the narrow section dimensions
section_length = 10  # Adjust as necessary
section_width = 5
section_height = 5

# Define obstacles
# Example: obstacle = {'center': [x, y, z], 'radius': r}
obstacles = [
    {'center': [2, 2, 2], 'radius': 0.5},
    {'center': [4, 3, 1], 'radius': 0.75},
    {'center': [6, 2, 3], 'radius': 0.5},
    # Add as many obstacles as needed
]

obstacles = []

# Parameters
path_start, path_end = 2, 8
horizontal_clearance = vertical_clearance = 0.20
obstacle_radius = 0.05  # Small radius to visualize the boundary
spacing = 1  # Distance between obstacles

# Generate obstacles along the tunnel boundaries
for x in np.arange(path_start, path_end + spacing, spacing):
    # Horizontal boundaries at y = +/- clearance
    obstacles.append({'center': [x, -horizontal_clearance, 0], 'radius': obstacle_radius})
    obstacles.append({'center': [x, horizontal_clearance, 0], 'radius': obstacle_radius})
    # Vertical boundaries at z = +/- clearance
    obstacles.append({'center': [x, 0, -vertical_clearance], 'radius': obstacle_radius})
    obstacles.append({'center': [x, 0, vertical_clearance], 'radius': obstacle_radius})

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, section_length)
ax.set_ylim(0, section_width)
ax.set_zlim(0, section_height)

# Draw each obstacle
for obs in obstacles:
    # For simplicity, this example uses scatter plot to represent obstacles
    # For a more accurate representation, consider using a custom function to draw spheres
    ax.scatter(*obs['center'], s=obs['radius']*1000, label='Obstacle')

plt.show()

 








