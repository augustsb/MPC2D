

import numpy as np
import matplotlib.pyplot as plt


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



"""
# Example usage
start = (0, 0)
goal = (13, 0)
obstacles = [
    {'center': (18.0, -1.0), 'radius': 2.0},  # o0
    {'center': (10.0, 0.0), 'radius': 1.5},   # o1
    {'center': (6.0, 1.0), 'radius': 1.0},    # o2
    {'center': (5.0, 4.0), 'radius': 2.0},    # o3
    {'center': (5.0, -4.0), 'radius': 2.0},   # o4
    {'center': (25.0, 4.0), 'radius': 2.0},   # o5
    {'center': (25.0, -4.0), 'radius': 2.0},  # o6
    {'center': (15.0, 4.0), 'radius': 1.0},   # o7
    {'center': (15.0, -4.0), 'radius': 1.0}   # o8
]

# Plotting for visualization
plt.figure()
for o in obstacles:
    circle = plt.Circle(o['center'], o['radius'], color='r', fill=True)
    plt.gca().add_patch(circle)

path = rrt(start, goal, obstacles, safe_margin)
N = 10  # Desired number of waypoints
path = interpolate_path(path, 13)

if path.any():
    xs, ys = zip(*path)
    plt.plot(xs, ys, '-o')
    plt.scatter(*zip(*[start, goal]), color=['green', 'red'])  # Start in green, goal in red
    plt.show()

"""

   