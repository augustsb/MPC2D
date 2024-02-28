import casadi
import numpy as np


def calculate_total_distance(optimized_path):
    # optimized_path is a 2D array with shape (N+1, 2), where N is the number of intervals
    distances = np.sqrt(np.sum(np.diff(optimized_path, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    return total_distance

def calculate_new_number_of_states(total_distance, desired_step_length):
    n = np.ceil(total_distance / desired_step_length)
    return n

def calculate_new_step_length(current_state, last_state, n):
    step_length = (last_state - current_state) / n
    return step_length

def generate_new_states(current_state, step_length, n):
    new_states = [current_state + i * step_length for i in range(n + 1)]
    return np.array(new_states)

def generate_initial_waypoints(start, goal, N):
    # Linearly interpolate between start and goal to generate N waypoints
    start= np.squeeze(start)  # This converts shape (2,1) to (2,)
    # Assuming target is already in the correct shape (2,), but if not:
    goal = np.squeeze(goal)  # Ensure target is also a 1D array of shape (2,)   
    waypoints = np.linspace(start, goal, N+1)  # Including both start and goal
    return waypoints


def redistribute_waypoints(optimized_path, k, p_CM):
    # Calculate distances between consecutive waypoints
    distances = np.sqrt(np.sum(np.diff(optimized_path, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    total_length = cumulative_distances[-1]
    num_waypoints = int(np.floor(total_length / k)) + 1

    # Generate evenly spaced points along the total length
    spaced_points = np.linspace(0, total_length, num_waypoints)
    
    # Interpolate x, y, and z coordinates separately
    new_xs = np.interp(spaced_points, cumulative_distances, optimized_path[:, 0])
    new_ys = np.interp(spaced_points, cumulative_distances, optimized_path[:, 1])
    new_zs = np.interp(spaced_points, cumulative_distances, optimized_path[:, 2])  # For 3D
    
    # Combine interpolated x, y, and z into new waypoints
    new_waypoints = np.vstack((new_xs, new_ys, new_zs)).T  # Updated for 3D

    return new_waypoints



def calculate_enhanced_safe_radius(obstacle_radius, snake_length):
    
    enhanced_safe_radius = obstacle_radius + (snake_length) / 2
    # Add an additional safety margin to define the safe distance
    additional_margin = 0
    safe_distance = enhanced_safe_radius + additional_margin
    return safe_distance


def repulsive_potential(x, y, obstacle, safe_distance):
    o_pos = obstacle['center']
    # Calculate the enhanced safe distance dynamically
    distance_to_obstacle = casadi.sqrt((x - o_pos[0])**2 + (y - o_pos[1])**2)
    
    # Use the safe_distance to determine when the repulsive potential is active
    potential = casadi.if_else(distance_to_obstacle < safe_distance, 
                               1 / casadi.fmax(distance_to_obstacle - obstacle['radius'], 1e-6), 
                               0)
    return potential



def calculate_clearance(current_p, obstacles):
    """
    Calculates the minimum distance from the current position to any obstacle,
    considering the obstacle's radius.

    Parameters:
    - current_p: The current position of the robot (numpy array or list of x, y coordinates).
    - obstacles: A list of dictionaries, each representing an obstacle with 'center' and 'radius'.
    - default_clearance: The default minimum clearance (safety distance) to maintain from obstacles.

    Returns:
    - The minimum clearance from the closest obstacle or the default clearance if the robot is not close to any obstacle.
    """
    min_clearance = float('inf')  # Start with infinity as the minimum clearance

    for obstacle in obstacles:
        # Extract obstacle center and radius
        o_center = obstacle['center']
        o_radius = obstacle['radius']

        # Calculate the Euclidean distance from the current position to the obstacle center
        distance_to_obstacle = np.linalg.norm(np.array(current_p) - np.array(o_center))

        # Calculate clearance by subtracting the obstacle's radius from the distance
        clearance = distance_to_obstacle - o_radius

        # Update min_clearance if this clearance is smaller
        if clearance < min_clearance:
            min_clearance = clearance

    # Ensure that the clearance is not less than the default minimum clearance
    return min_clearance


def calculate_curvature(waypoints):
    angles = []
    for i in range(1, len(waypoints)-1):
        a = waypoints[i-1]
        b = waypoints[i]
        c = waypoints[i+1]
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1, 1))
        angles.append(angle)
    max_angle = np.max(angles)
    total_curvature = np.sum(angles)
    return angles, total_curvature, max_angle

def calculate_distances(waypoints):
    if waypoints.ndim != 2 or waypoints.shape[0] < 2:
         raise ValueError("Waypoints array must be 2D with at least two waypoints.")
    distances = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    return distances, total_distance





def calculate_turning_angles(waypoints):
    def unit_vector(vector):
        """ Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2' """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return angle

    angles = []
    for i in range(1, len(waypoints)-1):
        prev_segment = waypoints[i] - waypoints[i-1]
        next_segment = waypoints[i+1] - waypoints[i]
        
        # Calculate the angle between the two segments
        angle = angle_between(prev_segment, next_segment)
        angles.append(angle)
    
    # Convert angles to degrees for easier interpretation
    angles_deg = np.degrees(angles)
    
    max_angle_deg = np.max(angles_deg)
    total_curvature_deg = np.sum(angles_deg)
    
    return angles_deg, total_curvature_deg, max_angle_deg
