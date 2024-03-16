import json
import numpy as np
import matplotlib.pyplot as plt


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

def draw_sphere(ax, center, radius, color='r'):
    # Generate points for a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.5,  zorder=2)



def visualize_simulation_results():
    # Path to your log file
    log_file_path = "simulation_log.json"

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    #middle_link_log = np.array(log_data['middle_link_log'])
    all_link_x_log = np.array(log_data['all_link_x'])
    all_link_y_log = np.array(log_data['all_link_y'])
    #all_link_z_log = np.array(log_data['all_link_z'])
    obstacles = log_data['obstacles']

    plt.figure(figsize=(10, 6))

    # Plot p_CM positions
    plt.plot(p_CM_log[:, 0], p_CM_log[:, 1], label='Path', marker='o', linestyle='-', markersize=0.1, color='b')

    # Plot middle link positions
    #plt.plot(middle_link_log[:, 0], middle_link_log[:, 1], label='Middle Link Path', marker='x', linestyle='-', markersize=0.1, color='y')

    for link_idx in range(all_link_x_log.shape[1]):  # Assuming second dimension is number of links
        x = np.squeeze(all_link_x_log[:, link_idx])  # Remove unnecessary dimension
        y = np.squeeze(all_link_y_log[:, link_idx])
        plt.plot(x, y, marker='x', linestyle='-', markersize=0.3, label=f'Link {link_idx}')

    # Plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle((obstacle['center'][0], obstacle['center'][1]), obstacle['radius'], color='red', fill=True, alpha=0.5)
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



def visualize_simulation_results_3d():

    # Path to your log file
    log_file_path = "simulation_log.json"


    waypoints = np.array([
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 1.04165649e+00,-6.74958878e-02,  4.06981388e-01],
                        [ 2.08332436e+00, -1.34991246e-01,  8.13968089e-01],
                        [ 3.12500511e+00, -2.02485676e-01,  1.22096136e+00],
                        [ 4.16670058e+00, -2.69978319e-01,  1.62796312e+00],
                        [ 5.20841362e+00, -3.37466198e-01,  2.03497692e+00],
                        [ 6.24967871e+00, -4.00503844e-01,  2.44390828e+00],
                        [ 7.28546506e+00, -4.16226337e-01,  2.87287802e+00],
                        [ 8.32124267e+00, -4.31924811e-01, 3.30184777e+00],
                        [ 9.35705847e+00, -4.47603353e-01,  3.73075776e+00],
                        [ 1.04183564e+01, -3.91362592e-01,  3.82288216e+00],
                        [ 1.13644360e+01,-2.27406968e-01, 3.35822765e+00],
                        [ 1.20000000e+01, -1.02576682e-31,  2.48275862e+00]
                        ])

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    #middle_link_log = np.array(log_data['middle_link_log'])
    all_link_x_log = np.array(log_data['all_link_x'])
    all_link_y_log = np.array(log_data['all_link_y'])
    all_link_z_log = np.array(log_data['all_link_z'])
    obstacles = log_data['obstacles']

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=40, azim=120)

    # Plot p_CM positions
    #ax.plot(p_CM_log[:, 0], p_CM_log[:, 1], p_CM_log[:, 2], label='Path', marker='o', linestyle='-', markersize=0.1, color='b')

    # Plot middle link positions
    #ax.plot(middle_link_log[:, 0], middle_link_log[:, 1], middle_link_log[:, 2], label='Middle Link Path', marker='x', linestyle='-', markersize=0.1, color='y')

    for link_idx in range(all_link_x_log.shape[1]):  # Assuming second dimension is number of links
        x = np.squeeze(all_link_x_log[:, link_idx])  # Remove unnecessary dimension
        y = np.squeeze(all_link_y_log[:, link_idx])
        z = np.squeeze(all_link_z_log[:, link_idx])
        ax.plot(x, y, z, marker='x', linestyle='-', markersize=0.3, label=f'Link {link_idx}', zorder=10)

    # Plot obstacles as spheres
    for obstacle in obstacles:
        # This part is tricky because matplotlib doesn't support 3D spheres directly
        # You could use plot_surface with a parametric sphere, but here's a simpler placeholder:
        #ax.scatter(obstacle['center'][0], obstacle['center'][1], obstacle['center'][2], color='red', s=100, label='Obstacles', alpha=0.5)
        draw_sphere(ax, obstacle['center'], obstacle['radius'], color='r')

    # Extracting X, Y, and Z coordinates

    #X, Y, Z = waypoints[:,0], waypoints[:,1], waypoints[:,2]

    # Plotting
    #ax.plot(X, Y, Z, marker='o', linestyle='-', color='b', markersize=5, label='Waypoints')


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


    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Snake Robot Simulation Paths in 3D')

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

