import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from casadi import MX, dot, fmax, fmin, sumsqr, sqrt

def is_too_close(point_a, point_b, min_distance):
    """
    Check if the distance between two points is less than the minimum allowed distance.
    """
    print(f"point_a shape: {point_a.shape}, point_b shape: {point_b.shape}")  # Add this line
    distance = np.linalg.norm(point_a - point_b)
    return distance < min_distance

def is_obstacle_too_close(new_obstacle, existing_obstacles):
    """
    Check if the new obstacle is too close to any existing obstacle.
    """
    for obstacle in existing_obstacles:
        # Calculate the minimum allowed distance between the obstacles (sum of their radii)
        min_allowed_distance = obstacle['radius'] + new_obstacle['radius']
        if is_too_close(new_obstacle['center'], obstacle['center'], min_allowed_distance):
            return True
    return False



def generate_random_obstacles(num_obstacles, area_size, start, target, min_distance_to_start_target):
    obstacles = []
    attempts = 0
    max_attempts = 100  # Increased limit on attempts for better obstacle placement

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        # Adjust center generation for the specified box area
        center_x = np.random.rand() * (area_size[0] - 0) + 0  # x=0 to x=31
        center_y = np.random.rand() * (2 * area_size[1]) - area_size[1]  # y=+-8
        center = np.array([center_x, center_y, 0])  # 3D point with z=0
        
        radius = np.random.rand() * (2 - 0.5) + 0.5  # Random radius between 0.5 and 2
        
        new_obstacle = {'center': center, 'radius': radius}

        # Check distances from the obstacle center to the start and target points
        start_too_close = np.linalg.norm(center[:2] - start[:2]) < (min_distance_to_start_target + radius)
        target_too_close = np.linalg.norm(center[:2] - target[:2]) < (min_distance_to_start_target + radius)
        other_obstacles_too_close = any(np.linalg.norm(center[:2] - o['center'][:2]) < (radius + o['radius']) for o in obstacles)

        if not start_too_close and not target_too_close and not other_obstacles_too_close:
            obstacles.append(new_obstacle)

    return obstacles




def visualize_obstacles(obstacles, area_size, start, target):
    fig, ax = plt.subplots()
    # Draw the box
    box = patches.Rectangle((0, -area_size[1]), area_size[0], 2*area_size[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box)
    
    # Plot start and target positions
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')  # Green dot
    ax.plot(target[0], target[1], 'ro', markersize=10, label='Target')  # Red dot

    # Draw obstacles
    for obstacle in obstacles:
        circle = patches.Circle((obstacle['center'][0], obstacle['center'][1]), obstacle['radius'], edgecolor='b', facecolor='blue', alpha=0.5)
        ax.add_patch(circle)
        #print(f"Obstacle at {obstacle['center'][:2]} with radius {obstacle['radius']}")

    plt.xlim(0, area_size[0])
    plt.ylim(-area_size[1], area_size[1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Obstacle Scenario')
    plt.grid(True)
    plt.show()

    # After showing the plot, print the obstacles in the structured format
    print("obstacles = [")
    for i, obstacle in enumerate(obstacles):
        center = obstacle['center']
        print(f"    {{'center': ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), 'radius': {obstacle['radius']:.2f}}},  # o{i}")
    print("]")




def calculate_min_dist_to_obstacle(A, B, obstacle_center, obstacle_radius):
    """
    CasADi-compatible version to calculate the minimum distance from a line segment AB to a circular obstacle.
    """
    # Ensure inputs are CasADi MX symbols
    A = MX(A)
    B = MX(B)
    obstacle_center = MX(obstacle_center)
    
    AB = B - A
    AO = obstacle_center - A
    AB_norm_sq = dot(AB, AB)
    AO_dot_AB = dot(AO, AB)
    
    # Project AO onto AB to find the closest point (in a CasADi-compatible way)
    t = AO_dot_AB / AB_norm_sq
    
    # Ensure t is within [0, 1] using CasADi fmax and fmin functions
    t_clamped = fmax(0, fmin(1, t))
    
    # Calculate the closest point on AB to the obstacle center
    closest_point = A + t_clamped * AB
    
    # Calculate the squared distance from the closest point to the obstacle center
    dist_sq_to_center = sumsqr(closest_point - obstacle_center)
    dist_to_edge = sqrt(dist_sq_to_center) - obstacle_radius
    
    return dist_to_edge



def filter_obstacles(current_p, obstacles, horizon_distance):
    """
    Filter obstacles to include only those within a certain horizon distance.

    :param current_p: Current position of the robot (numpy array).
    :param obstacles: List of obstacles, each described by a dictionary with 'center' and 'radius'.
    :param horizon_distance: Distance defining the horizon within which obstacles are considered.
    :return: Filtered list of obstacles within the horizon.
    """
    filtered_obstacles = []
    for obstacle in obstacles:
        o_pos = np.array(obstacle['center'])
        distance_to_obstacle = np.linalg.norm(o_pos - current_p)
        if distance_to_obstacle <= horizon_distance:
            filtered_obstacles.append(obstacle)
    return filtered_obstacles