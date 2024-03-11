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

def visualize_simulation_results():
    # Path to your log file
    log_file_path = "simulation_log.json"

    # Read the data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    p_CM_log = np.array(log_data['p_CM_log'])
    middle_link_log = np.array(log_data['middle_link_log'])
    obstacles = log_data['obstacles']

    plt.figure(figsize=(10, 6))

    # Plot p_CM positions
    plt.plot(p_CM_log[:, 0], p_CM_log[:, 1], label='Path', marker='o', linestyle='-', markersize=0.1, color='b')

    # Plot middle link positions
    plt.plot(middle_link_log[:, 0], middle_link_log[:, 1], label='Middle Link Path', marker='x', linestyle='-', markersize=0.1, color='y')

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