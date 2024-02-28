import numpy as np

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
    max_attempts = num_obstacles * 10  # Set a limit on attempts to prevent infinite loop
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        center_2d = np.random.rand(2) * np.array(area_size)
        # Extend the 2D center to a 3D point with z coordinate set to 0
        center = np.append(center_2d, 0)  # Now center is a 3D point with z=0
        
        radius = np.random.rand() * (1.5 - 0.3) + 0.3  # Random radius between 0.5 0.1
        new_obstacle = {'center': center, 'radius': radius}

        # Check distances from the obstacle center to the start and target points
        start_too_close = is_too_close(center, start, min_distance_to_start_target + radius)
        target_too_close = is_too_close(center, target, min_distance_to_start_target + radius)
        other_obstacles_too_close = is_obstacle_too_close(new_obstacle, obstacles)  # Renamed variable

        if not start_too_close and not target_too_close and not other_obstacles_too_close:
            obstacles.append(new_obstacle)

    return obstacles
