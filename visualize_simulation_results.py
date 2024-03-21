import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # For 3D line collections
from scipy.stats import iqr



def convert_to_serializable(item):
    """Recursively convert numpy arrays, CasADi DM, and other non-serializable items to serializable formats."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    elif str(type(item)).endswith("casadi.casadi.DM'>"):
        return np.array(item).tolist()  # Convert DM to numpy array, then to list
    elif isinstance(item, dict):
        return {key: convert_to_serializable(val) for key, val in item.items()}
    elif isinstance(item, list):
        return [convert_to_serializable(val) for val in item]
    else:
        return item


import os



def plot_colored_path(mode, r_acceptance = 0.5):
    # Read the data
    dt = 0.05

    log_file_path = "simulation_log.json"
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    velocity_log = np.array(log_data['velocity_list'])  # Assuming this is correct; remove extra list wrap if necessary
    energy_log = np.array(log_data['energy_list'])  # Same as above
    energy_per_second = np.array(energy_log) / dt  # Adjust energy values to per-second basis if necessary
    obstacles = log_data['obstacles']

    avg_solver_time = log_data['avg_solver_time']
    avg_power_usage = log_data['avg_power_usage']
    total_distance_traveled = log_data['tot_distance']
    total_time = log_data['tot_time']
    target = log_data["target"]


    #target_array = np.array(target)
    #print(target_array.shape)
    #print(p_CM_log[-1, :].shape)
    #print("The norm between the last position and the target is:", np.linalg.norm(p_CM_log[-1, :].flatten() - target_array))


    power_iqr = iqr(energy_per_second)
    power_median = np.median(energy_per_second)
    vmin = max(power_median - 0.5 * power_iqr, min(energy_per_second))
    vmax = min(power_median + 0.5 * power_iqr, max(energy_per_second))

    text_str_power = f"Average Power Usage: {avg_power_usage:.2f} W"
    text_str_velocity = f"Average Speed: {total_distance_traveled / total_time:.2f} m/s"


    # Function to plot the paths
    def plot_path_with_color(p_CM_log, values, title, text_str, cbar_label, fig_path, vmin, vmax):
        plt.figure(figsize=(10, 6))
        x = p_CM_log[:, 0]
        y = p_CM_log[:, 1]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection object
        lc = LineCollection(segments, cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        lc.set_array(values)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.colorbar(lc, label=cbar_label)

        # Plot obstacles
        for obstacle in obstacles:
            circle = plt.Circle((obstacle['center'][0], obstacle['center'][1]), obstacle['radius'], color='red', fill=True, alpha=0.5)
            plt.gca().add_patch(circle)

        plt.plot(x[0], y[0], 'go', markersize=7, label='Start')
        plt.plot(x[-1], y[-1], 'bo', markersize=7, label='End')

        plt.xlabel('X[m]')
        plt.ylabel('Y[m]')
        #plt.title(title)
        plt.axis('equal')
        plt.legend()
        plt.grid(True)

        if not os.path.exists('figs'):
            os.makedirs('figs')
        #text_str = f"Average Solver Time: {avg_solver_time:.4f} s\nAverage Power Usage: {avg_power_usage:.2f} W\nAverage Speed: {total_distance_traveled / total_time:.2f} m/s"
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")
        plt.show()


    def plot_path_with_color_3d(p_CM_log, values, title, text_str, cbar_label, fig_path, vmin, vmax):

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = p_CM_log[:, 0], p_CM_log[:, 1], p_CM_log[:, 2]

        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        lc.set_array(values)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        #cbar = fig.colorbar(lc, ax=ax, label=cbar_label)
        cbar = fig.colorbar(lc, ax=ax, label=cbar_label, shrink=0.5, aspect=20, pad=0.02)

        for obstacle in obstacles:
            draw_sphere(ax, obstacle['center'], obstacle['radius'], color='k')

        ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=7, label='Start')
        #ax.plot([x[-1]], [y[-1]], [z[-1]], 'bo', markersize=7, label='End')
        ax.plot([target[0]], [target[1]], [target[2]], 'bo', markersize=2, label='Goal')

        draw_sphere_acceptance(ax, target, r_acceptance, color='blue', alpha=0.5, linewidth=0.1)

        ax.set_xlabel('X[m]')
        ax.set_ylabel('Y[m]')
        ax.set_zlabel('Z[m]')
        #ax.set_title(title)
        ax.set_xlim(0, 29)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 6)
        plt.legend()
        plt.grid(True)

        if not os.path.exists('figs'):
            os.makedirs('figs')

        #text_str = f"Average Solver Time: {avg_solver_time:.4f} s\nAverage Power Usage: {avg_power_usage:.2f} W\nAverage Speed: {total_distance_traveled / total_time:.2f} m/s"
        ax.text2D(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))
        set_axes_equal(ax)
        # For 3D plotting, use this function after all other plotting commands and just before plt.show() or plt.savefig():

        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")
        plt.show()


    if (mode == '3D'):

        plot_path_with_color_3d(p_CM_log, velocity_log, 'CM Path with Velocity Coloring', text_str_velocity, 'Velocity [m/s]', 'figs/cm_path_velocity.png', 0, 0.6)
        plot_path_with_color_3d(p_CM_log, energy_per_second, 'CM Path with Energy Usage Coloring', text_str_power, 'Energy Usage [W]', 'figs/cm_path_energy.png', 5, 30)

    else:
        plot_path_with_color(p_CM_log, velocity_log, 'CM Path with Velocity Coloring', text_str_velocity, 'Velocity [m/s]', 'figs/cm_path_velocity.png', min(velocity_log), max(velocity_log))
        plot_path_with_color(p_CM_log, energy_per_second, 'CM Path with Energy Usage Coloring', text_str_power, 'Energy Usage [W]', 'figs/cm_path_energy.png', vmin, vmax)




def set_axes_equal(ax):
    # Extract the limits for the X, Y, and Z axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the ranges and find the maximum range
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    # Calculate the midpoints for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    # Set the limits for each axis to the maximum range
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)



def draw_sphere(ax, center, radius, color='r'):
    # Generate points for a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.5)


def draw_sphere_acceptance(ax, center, radius, **kwargs):
    # Generate points on a sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # Higher resolution
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    # Plot the points as a surface
    ax.plot_surface(x, y, z, **kwargs)

def distance_to_obstacle_edge(point, obstacle):
    # Extract obstacle center and radius
    obstacle_center = np.array(obstacle['center']).flatten()  # Flatten the obstacle center array
    radius = obstacle['radius']
    
    # Ensure point is a flat array to match obstacle_center's dimensions
    point = np.array(point).flatten()
    
    # Calculate Euclidean distance from point to obstacle center
    distance_center_to_point = np.linalg.norm(point[:2] - obstacle_center[:2])  # Consider only X and Y for 2D distance
    
    # Calculate distance from point to the nearest edge of the obstacle
    distance_to_edge = distance_center_to_point
    return distance_to_edge


def visualize_simulation_results():
    # Path to your log file
    log_file_path = "simulation_log.json"

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    phi_ref_x_log = np.array(log_data['phi_ref_x'])
    velocity_log = np.array([log_data['velocity_list']])
    energy_log = np.array([log_data['energy_list']])
    #middle_link_log = np.array(log_data['middle_link_log'])
    all_link_x_log = np.array(log_data['all_link_x'])
    all_link_y_log = np.array(log_data['all_link_y'])
    #all_link_z_log = np.array(log_data['all_link_z'])
    obstacles = log_data['obstacles']

    plt.figure(figsize=(10, 6))
  

    min_distances = []

    # Iterate over each point in p_CM_log
    for p_CM in p_CM_log:
        # Calculate distances to all obstacles and pick the minimum
        distances = [distance_to_obstacle_edge(p_CM[:2], obstacle) for obstacle in obstacles]
        min_distance = min(distances)
        min_distances.append(min_distance)

    min_distances = np.array(min_distances)


    """
    min_distances = []

    # Assume obstacles is a list of dictionaries, each with 'center' and 'radius' keys
    for time_step in range(len(all_link_x_log)):
        link_distances_at_timestep = []
        for link_idx in range(all_link_x_log.shape[1]):  # Iterate over each link
            link_position = np.array([all_link_x_log[time_step, link_idx], all_link_y_log[time_step, link_idx]])
            # Calculate distances from this link to all obstacles and pick the minimum
            distances = [distance_to_obstacle_edge(link_position, obstacle) for obstacle in obstacles]
            link_distances_at_timestep.append(min(distances))
        # Store the minimum distance among all links for this time step
        min_distances.append(min(link_distances_at_timestep))

    min_distances = np.array(min_distances)

    """


    # Plot p_CM positions
    #plt.plot(p_CM_log[:, 0], p_CM_log[:, 1], label='Path', marker='o', linestyle='-', markersize=0.1, color='b')

    # Plot middle link positions
    #plt.plot(middle_link_log[:, 0], middle_link_log[:, 1], label='Middle Link Path', marker='x', linestyle='-', markersize=0.1, color='y')

    for link_idx in range(all_link_x_log.shape[1]):  # Assuming second dimension is number of links
        x = np.squeeze(all_link_x_log[:, link_idx])  # Remove unnecessary dimension
        y = np.squeeze(all_link_y_log[:, link_idx])
        #plt.plot(x, y, marker='x', linestyle='-', markersize=0.3, label=f'Link {link_idx}')
        plt.plot(x, y, marker='x', linestyle='-', markersize=0.3)

    # Plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle((obstacle['center'][0], obstacle['center'][1]), obstacle['radius'], color='red', fill=True, alpha=1.0)
        plt.gca().add_patch(circle)

    plt.plot([], [], 'o', color='red', label='Obstacles', alpha=0.5)
    # Plot start and end points
    start_point = p_CM_log[0]
    end_point = p_CM_log[-1]
    plt.plot(start_point[0], start_point[1], 'go', markersize=7, label='Start')
    plt.plot(end_point[0], end_point[1], 'bo', markersize=7, label='End')

    plt.xlabel('X[m]')
    plt.ylabel('Y[m]')
    plt.title('Snake Robot Simulation Paths')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Check if the 'figs' directory exists, create it if not
    if not os.path.exists('figs'):
        os.makedirs('figs')

    # Save the figure to the 'figs' directory
    figure_path = os.path.join('figs', 'simulation_results.png')
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")

    # Optionally, you can still display the plot in addition to saving it
    #plt.show()

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(len(phi_ref_x_log))

    # Plotting phi_ref_x
    plt.plot(time_steps, phi_ref_x_log, label='$\phi_{ref\_x}$ over time', color='green', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('$\phi_{ref\_x}$ (Radians)', color='green')
    plt.tick_params(axis='y', labelcolor='green')

    # Create a second y-axis for the minimum distance to the closest obstacle
    ax2 = plt.twinx()
    ax2.plot(time_steps, min_distances, label='Distance to Closest Obstacle', color='blue', linestyle='--')
    ax2.set_ylabel('Distance (m)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('Reference Angular Position and Distance to Closest Obstacle Over Time')
    plt.grid(True)

    # Save the combined figure
    combined_figure_path = os.path.join('figs', 'phi_ref_x_and_distance_over_time.png')
    plt.savefig(combined_figure_path)
    print(f"Combined figure saved to {combined_figure_path}")
    plt.show()



def draw_snake_robot_at_time_t(idx, ax,  l = 0.07, n = 10):

    log_file_path = "simulation_log.json"

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    #middle_link_log = np.array(log_data['middle_link_log'])
    all_link_x_log = np.array(log_data['all_link_x'])
    all_link_y_log = np.array(log_data['all_link_y'])
    all_link_z_log = np.array(log_data['all_link_z'])
    theta_x_list = np.array(log_data['theta_x_list'])
    theta_z_list = np.array(log_data['theta_z_list'])
    target = log_data["target"]

    x = all_link_x_log[idx]
    y = all_link_y_log[idx]
    z = all_link_z_log[idx]
    theta_x = theta_x_list[idx]
    theta_z = theta_z_list[idx]


    for j in range(n):

        startx = x[j] - l*np.cos(theta_z[j])*np.cos(theta_x[j])
        starty = y[j] - l*np.cos(theta_z[j])*np.sin(theta_x[j])
        startz = z[j] + l*np.sin(theta_z[j])
        endx = x[j] + l*np.cos(theta_z[j])*np.cos(theta_x[j])
        endy = y[j] + l*np.cos(theta_z[j])*np.sin(theta_x[j])
        endz = z[j] - l*np.sin(theta_z[j])

        startx = np.squeeze(startx)
        starty = np.squeeze(starty)
        startz = np.squeeze(startz)
        endx = np.squeeze(endx)
        endy = np.squeeze(endy)
        endz = np.squeeze(endz)

        # Plotting the links
        ax.plot([startx, endx], [starty, endy], [startz, endz], 'bo-', linewidth=1.0, zorder=20, markersize=3)
 


def visualize_simulation_results_3d(r_acceptance=0.5):

    dt = 0.05

    # Path to your log file
    log_file_path = "simulation_log.json"

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    #middle_link_log = np.array(log_data['middle_link_log'])
    all_link_x_log = np.array(log_data['all_link_x'])
    all_link_y_log = np.array(log_data['all_link_y'])
    all_link_z_log = np.array(log_data['all_link_z'])
    obstacles = log_data['obstacles']

    n = 25  # Adjust n to change the sparsity
    sparse_p_CM_log = p_CM_log[::n, :]



    velocity_log = np.array(log_data['velocity_list'])  # Assuming this is correct; remove extra list wrap if necessary
    energy_log = np.array(log_data['energy_list'])  # Same as above
    energy_per_second = np.array(energy_log) / dt  # Adjust energy values to per-second basis if necessary
    obstacles = log_data['obstacles']

    avg_solver_time = log_data['avg_solver_time']
    avg_power_usage = log_data['avg_power_usage']
    total_distance_traveled = log_data['tot_distance']
    total_time = log_data['tot_time']
    target = log_data["target"]



    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=40, azim=120)

    # Plot p_CM positions
    #ax.plot(p_CM_log[:, 0], p_CM_log[:, 1], p_CM_log[:, 2], label='Path', marker='o', linestyle='-', markersize=0.1, color='b')
    #ax.plot(p_CM_log[:, 0], p_CM_log[:, 1], p_CM_log[:, 2], marker='x', linestyle='-', markersize=0.3, zorder=10)
    ax.scatter(sparse_p_CM_log[:, 0], sparse_p_CM_log[:, 1], sparse_p_CM_log[:, 2], c='b', marker='o', s=5)
    # Plot middle link positions
    #ax.plot(middle_link_log[:, 0], middle_link_log[:, 1], middle_link_log[:, 2], label='Middle Link Path', marker='x', linestyle='-', markersize=0.1, color='y')

    """

    for link_idx in range(all_link_x_log.shape[1]):  # Assuming second dimension is number of links
        x = np.squeeze(all_link_x_log[:, link_idx])  # Remove unnecessary dimension
        y = np.squeeze(all_link_y_log[:, link_idx])
        z = np.squeeze(all_link_z_log[:, link_idx])
        ax.plot(x, y, z, marker='x', linestyle='-', markersize=0.3, label=f'Link {link_idx}', zorder=10)
    """

    # Plot obstacles as spheres
    for obstacle in obstacles:
        draw_sphere(ax, obstacle['center'], obstacle['radius'], color='r')

    
    ax.plot([target[0]], [target[1]], [target[2]], 'bo', markersize=7, label='Goal')
    plt.plot(p_CM_log[0, 0], p_CM_log[0,0], 'go', markersize=7, label='Start')
    plt.plot([], [], 'o', color='red', label='Obstacles', alpha=1.0)
    draw_sphere_acceptance(ax, target, r_acceptance, color='blue', alpha=0.5, linewidth=0.1)


    draw_snake_robot_at_time_t(200, ax,  l = 0.07, n = 10)

    draw_snake_robot_at_time_t(500, ax,  l = 0.07, n = 10)

    draw_snake_robot_at_time_t(700, ax,  l = 0.07, n = 10)

    draw_snake_robot_at_time_t(1000, ax,  l = 0.07, n = 10)

    draw_snake_robot_at_time_t(1300, ax,  l = 0.07, n = 10)


    # Manually set the aspect ratio
    max_range = np.array([p_CM_log[:, 0].max()-p_CM_log[:, 0].min(), 
                          p_CM_log[:, 1].max()-p_CM_log[:, 1].min(), 
                          p_CM_log[:, 2].max()-p_CM_log[:, 2].min()]).max() / 2.0

    mid_x = (p_CM_log[:, 0].max()+p_CM_log[:, 0].min()) * 0.5
    mid_y = (p_CM_log[:, 1].max()+p_CM_log[:, 1].min()) * 0.5
    mid_z = (p_CM_log[:, 2].max()+p_CM_log[:, 2].min()) * 0.5

    ax.set_xlim(0, 29)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    # Calculate the focus area, e.g., the midpoint of the robot's path or a specific time step
    focus_midpoint = np.mean(p_CM_log, axis=0)

    # Define the zoom range - the distance around the midpoint you want to include in the view
    zoom_range = 10.0  # Adjust this value to zoom in or out

    # Set axis limits based on the focus_midpoint and zoom_range
    ax.set_xlim(focus_midpoint[0] - zoom_range, focus_midpoint[0] + zoom_range)
    ax.set_ylim(focus_midpoint[1] - zoom_range, focus_midpoint[1] + zoom_range)
    ax.set_zlim(focus_midpoint[2] - zoom_range, focus_midpoint[2] + zoom_range)


    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    #ax.set_title('Snake Robot Path All Links')

    plt.legend()
    plt.grid(True)

    # Check if the 'figs' directory exists, create it if not
    if not os.path.exists('figs'):
        os.makedirs('figs')

    # Save the figure to the 'figs' directory
    figure_path = os.path.join('figs', 'simulation_results_3d.png')
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")

    # Show plot
    plt.show()



#visualize_simulation_results_3d()
visualize_simulation_results() 

#plot_colored_path('3D')