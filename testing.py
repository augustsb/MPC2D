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
#target = np.array([29.0 , 0.0, 4.0]) #3D
target = np.array([29.0 , 0.0, 0.0]) #2D
start = np.array([0.0 , 0.0, 0.0]) #3D
obstacles = [
            {'center': (18.0, -1.0, 4.5), 'radius': 2.0},  # o0
            {'center': (10.0, 0.0, 1.8), 'radius': 1.5},   # o1
            {'center': (6.0, 1.0, 3.0), 'radius': 1.0},    # o2
            {'center': (5.0, 4.0, 4.0), 'radius': 2.0},    # o3
            {'center': (5.0, -4.0, 4.0), 'radius': 2.0},   # o4
            {'center': (25.0, 4.0, 4.0), 'radius': 2.0},   # o5
            {'center': (25.0, -4.0, 4.0), 'radius': 2.0},  # o6
            {'center': (15.0, 4.0, 2.0), 'radius': 1.0},   # o7
            {'center': (15.0, -4.0, 2.0), 'radius': 1.0}   # o8
]

obstacles_diagonal = [
    {'center': (4, -2, 1), 'radius': 1},
    {'center': (7, -1, 2), 'radius': 1},
    {'center': (10, 1, 3), 'radius': 1},
    {'center': (13, 2, 2), 'radius': 1},
    {'center': (16, 1, 3), 'radius': 1},
    {'center': (19, -1, 2), 'radius': 1},
    {'center': (22, -2, 1), 'radius': 1},
    {'center': (25, 0, 3), 'radius': 1},
]


obstacles_labyrinth = [
    # First row of obstacles, directly in line but with varying heights
    {'center': (5, 0, 1), 'radius': 2},
    {'center': (5, 0, 4), 'radius': 2},
    # Second row of obstacles, forcing a lateral maneuver
    {'center': (12, -3, 2), 'radius': 2},
    {'center': (12, 3, 4), 'radius': 2},
    # Third row, a tighter squeeze, height variance requires vertical maneuvering
    {'center': (19, -1, 1), 'radius': 1.5},
    {'center': (19, 1, 5), 'radius': 1.5},
    # Final challenge before the goal, close placement requires precise navigation
    {'center': (25, 0, 3), 'radius': 1},
    {'center': (27, 0, 2), 'radius': 1},
]


obstacles_serpentine_barrier = [
    # Stagger obstacles across the direct path, forcing a serpentine route
    {'center': (6, 1, 2), 'radius': 1.5},
    {'center': (9, -2, 3), 'radius': 1.5},
    {'center': (12, 2, 1), 'radius': 1.5},
    {'center': (15, -1, 3), 'radius': 1.5},
    {'center': (18, 1, 2), 'radius': 1.5},
    {'center': (21, -2, 1), 'radius': 1.5},
    {'center': (24, 2, 3), 'radius': 1.5},
    {'center': (27, -1, 2), 'radius': 1.5},
]

#2D

obstacles_zigzag_gate = [
    {'center': (5, 2), 'radius': 1},
    {'center': (10, -2), 'radius': 1},
    {'center': (15, 2), 'radius': 1},
    {'center': (20, -2), 'radius': 1},
    {'center': (25, 2), 'radius': 1},
]

obstacles_spiral =  [
    {'center': (4, 0), 'radius': 1},
    {'center': (7, 1), 'radius': 1},
    {'center': (10, -1), 'radius': 1},
    {'center': (13, 2), 'radius': 1},
    {'center': (16, -2), 'radius': 1},
    {'center': (19, 3), 'radius': 1},
    {'center': (22, -3), 'radius': 1},
    {'center': (20, 0), 'radius': 1},
    {'center': (25, 0), 'radius': 1},
]

obstacles_labyrinth = [
    # Outer walls (simplified for challenge)
    {'center': (5, 5), 'radius': 0.5},
    {'center': (5, -5), 'radius': 0.5},
    {'center': (10, 5), 'radius': 0.5},
    {'center': (10, -5), 'radius': 0.5},
    {'center': (15, 5), 'radius': 0.5},
    {'center': (15, -5), 'radius': 0.5},
    {'center': (20, 5), 'radius': 0.5},
    {'center': (20, -5), 'radius': 0.5},
    
    # Inner obstacles creating the labyrinth paths
    {'center': (7, 0), 'radius': 0.5},
    {'center': (9, 2), 'radius': 0.5},
    {'center': (9, -2), 'radius': 0.5},
    {'center': (11, 3), 'radius': 0.5},
    {'center': (11, -3), 'radius': 0.5},
    {'center': (13, 1), 'radius': 0.5},
    {'center': (13, -1), 'radius': 0.5},
    {'center': (17, 4), 'radius': 0.5},
    {'center': (17, -4), 'radius': 0.5},
    {'center': (19, 0), 'radius': 0.5},
    {'center': (21, 2), 'radius': 0.5},
    {'center': (21, -2), 'radius': 0.5},
    {'center': (23, 3), 'radius': 0.5},
    {'center': (23, -3), 'radius': 0.5},
    {'center': (25, 1), 'radius': 0.5},
    {'center': (25, -1), 'radius': 0.5},
    {'center': (27, 0), 'radius': 0.5}  # The final obstacle before the goal
]



obstacles = obstacles_maze_of_rings
# Plotting
fig, ax = plt.subplots()
#ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 29)
ax.set_ylim(-10, 10)
#ax.set_zlim(0, 6)


ax.plot([start[0]], [start[1]], 'go', markersize=7, label='Start')

ax.plot([target[0]], [target[1]], 'bo', markersize=2, label='Goal')

ax.plot([start[0], target[0]], [start[1], target[1-+]], 'r--', linewidth=2, label='Direct Path')
# Draw each obstacle
for obs in obstacles:
    # For simplicity, this example uses scatter plot to represent obstacles
    # For a more accurate representation, consider using a custom function to draw spheres
    ax.scatter(*obs['center'], s=obs['radius']*1000, label='Obstacle')

plt.show()

 








