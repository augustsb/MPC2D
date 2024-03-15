import numpy as np
import os
import pandas as pd


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
            if (np.linalg.norm(waypoints[i, :] - o_pos) < o_rad + safe_margin):
                collision = True
                break

            #if (i < num_path_states - 1 ):  # Clearance constraints for midpoints
                #midpoint = (waypoints[i,:] + waypoints[i+1, :]) / 2
                #if (np.linalg.norm( midpoint- o_pos) < o_rad + safe_margin):
                  #  collision = True
                   # break  
        if collision:
            additional_steps += 1  # Increment if collision detected

        else:
            break  # Stop extending the horizon if no collision is detected
    # Adjust N based on the additional steps needed for obstacle avoidance
    N += additional_steps

    if (N > num_path_states):
        N = num_path_states

    return N


def expand_initial_guess(P_sol, N_new, goal):
    # Assuming P_sol is a 2D array with shape [num_states, state_dimension]
    # and N_new > P_sol.shape[0]
    P_sol = np.array(P_sol)
    # Copy existing states
    expanded_P_sol = np.zeros((N_new, P_sol.shape[1]))

    # Check if P_sol has more states than needed, which shouldn't normally happen but is good to check
    if P_sol.shape[0] > N_new:
        return P_sol[:N_new, :]
    
    # Copy existing states from P_sol
    num_existing_states = min(P_sol.shape[0], N_new)
    expanded_P_sol[:num_existing_states, :] = P_sol[:num_existing_states, :]

    # If additional states are needed, generate them
    if N_new > P_sol.shape[0]:
        # Calculate step for linear interpolation based on the difference between the last state of P_sol and the goal
        step = (goal - P_sol[-1, :]) / (N_new - P_sol.shape[0])
        for i in range(1, N_new - P_sol.shape[0] + 1):
            expanded_P_sol[P_sol.shape[0] + i - 1, :] = P_sol[-1, :] + step * i

    return expanded_P_sol





#NOT USED
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



def distance(a, b):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))


def line_intersects_circle(start, end, center, radius):
    """
    Check if the line segment between start and end intersects with the circle.
    This function assumes start, end, and center are NumPy arrays.
    """
    start, end, center = map(np.array, (start, end, center))
    line_vec = end - start
    center_to_start_vec = start - center
    a = np.dot(line_vec, line_vec)
    b = 2 * np.dot(center_to_start_vec, line_vec)
    c = np.dot(center_to_start_vec, center_to_start_vec) - radius**2
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        # No real roots, line does not intersect circle
        return False
    
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    if (0 <= t1 <= 1 or 0 <= t2 <= 1):
        # At least one intersection point is on the line segment
        return True
    
    return False


def is_collision_free(new_node, nearest_node, obstacles, safe_margin):
    """
    Checks if the path between new_node and nearest_node is free of collisions with any obstacles.
    """
    for o in obstacles:
        o_pos = np.array(o['center'])  # Obstacle position
        o_rad = o['radius'] + safe_margin  # Obstacle radius plus a safety margin
        
        if line_intersects_circle(np.array(nearest_node), np.array(new_node), o_pos, o_rad):
            # If the path intersects with the obstacle, it's not collision-free
            return False
            
    # If no collisions are detected with any obstacle, the path is considered collision-free
    return True


def rrt(start, goal, obstacles, safe_margin, iterations=10000, step_size=0.5):
    tree = {start: None}  # Dictionary to store the tree. Format: {node: parent_node}

    for _ in range(iterations):
        sampled_point = (np.random.uniform(low=0, high=30), np.random.uniform(low=-10, high=10), 0)  # 3D with z=0
        nearest_node = min(tree.keys(), key=lambda node: distance(node, sampled_point))
        
        # Create a new node towards sampled_point from nearest_node
        direction = np.array(sampled_point) - np.array(nearest_node)
        direction = direction / np.linalg.norm(direction) * step_size
        new_node = np.array(nearest_node) + direction
        new_node = tuple(new_node)  # Convert to tuple to use as a dictionary key
        
        if is_collision_free(new_node, nearest_node, obstacles, safe_margin):
            tree[new_node] = nearest_node  # Add the new node to the tree
            
            if distance(new_node, goal) <= step_size:  # Check if goal is reached
                print("Goal reached!")
                # Build and return the path from start to goal
                path = [new_node]
                while path[-1] != start:
                    path.append(tree[path[-1]])
                path.reverse()
                return path
    return None  # If goal not reached within iterations

def interpolate_path(path, N):
    """
    Interpolates a given path to produce exactly N waypoints.
    path: List of waypoints [(x1, y1), (x2, y2), ...]
    N: Desired number of waypoints
    """
    # Convert path to an array for easier manipulation
    path_array = np.array(path)
    # Calculate the total distance of the path
    distances = np.sqrt(((path_array[1:] - path_array[:-1])**2).sum(axis=1))
    total_distance = distances.sum()
    # Distance between each waypoint in the new path
    step_distance = total_distance / (N - 1)
    
    new_path = [path_array[0]]
    accumulated_distance = 0
    for i in range(1, len(path)):
        segment_distance = distances[i-1]
        while accumulated_distance + segment_distance >= step_distance:
            # Calculate interpolation ratio for the next waypoint
            ratio = (step_distance - accumulated_distance) / segment_distance
            # Interpolate and add the new waypoint
            new_waypoint = path_array[i-1] + ratio * (path_array[i] - path_array[i-1])
            new_path.append(new_waypoint)
            accumulated_distance = 0  # Reset accumulated distance
            path_array[i-1] = new_waypoint  # Update starting point for the next segment
            segment_distance = np.linalg.norm(path_array[i] - new_waypoint)
        accumulated_distance += segment_distance
    
    if len(new_path) < N:
        new_path.append(path_array[-1])
    
    return np.array(new_path)









