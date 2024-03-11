import casadi
import numpy as np


def generate_initial_path(x0, xgoal, k_abs):
    """
    Updates the initial path from the current state to the goal.

    Parameters:
    - x0: The current state of the vehicle (x, y, z).
    - xgoal: The goal state (x, y, z).
    - k_abs: The desired Euclidean step length between path states.

    Returns:
    - P: The updated path as an array of states (x, y, z).
    """
    # Compute the distance vector from the current state to the goal
    distance_vector = xgoal - x0
    
    # Compute the number of states between the current state and the goal
    n = int(np.linalg.norm(distance_vector) / k_abs) + 1
    
    # Compute the new step length vector
    if n > 1:
        k = distance_vector / (n - 1)
    else:
        # Handle the case where the current state is the goal or very close to it
        k = distance_vector
    
    # Generate the new initial path
    P = np.array([x0 + i * k for i in range(n)])
    
    return P


def path_resolution(x, x_opt, k_abs, N):
    """
    Updates the path resolution based on the optimized path and desired step length.

    Parameters:
    - x: States from the initial path
    - x_opt: Array of locally optimal states, shape (N, 3), where N is the number of states.
    - k_abs: The desired Euclidean step length between path states.
    - N: The number of states in the prediction horizon.

    Returns:
    - P: The updated path as an array of states.
    """

    # Calculate the Euclidean distance of the optimal path (lf)
    lf = sum(np.linalg.norm(x_opt[:,i+1] - x_opt[:, i]) for i in range(N-1))
    
    # Calculate the new number of states needed (n)
    n = int(lf / k_abs) 
    
    # Calculate the new step length vector (k)

    x0 = x[0]
    xN = x[N-1]

    if n > 0:
        k = (xN - x0) / n
    else:
        # Handle the case where n is 0, which can occur if lf < k_abs
        k = xN - x0
        n = 1  # Ensure at least one step

    # Generate the new initial path using the new step length
    P = np.array([x0 + i * k for i in range(n + 1)])

    """
    path_prev_index = x.shape[0]
    path_index = P.shape[0]
    diff = path_index - path_prev_index

    if (N + diff >= 10):
        N = N + diff
    """

    return P


def extend_horizon(waypoints, N, obstacles, num_path_states, controller_params):
    # Initialize the number of additional steps needed due to obstacles
    additional_steps = 0
    safe_margin = controller_params['alpha_h']

    N = 10

    # Start checking from the current horizon N
    for i in range(N-1, num_path_states):
        collision = False
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            #midpoint = (waypoints[i,:] + waypoints[i+1, :]) / 2

            # Check if the distance to the obstacle is less than or equal to its radius
            if np.linalg.norm(waypoints[i, :] - o_pos) <= o_rad + safe_margin:
                collision = True
                break

            if (i < num_path_states - 1 ):  # Clearance constraints for midpoints
                midpoint = (waypoints[i,:] + waypoints[i+1, :]) / 2
                if (np.linalg.norm( midpoint- o_pos) <= o_rad + safe_margin):
                    collision = True
                    break  
        if collision:
            additional_steps += 1  # Increment if collision detected

        else:
            break  # Stop extending the horizon if no collision is detected
    # Adjust N based on the additional steps needed for obstacle avoidance
    N += additional_steps

    if (N > num_path_states):
        N = num_path_states

    return N




