from casadi import SX, DM, vertcat
from calculate_diagonal_matrices import calculate_cos_theta_diag, calculate_sin_theta_diag

def calculate_XYZ_dot(theta_x_dot, theta_z_dot, p_CM_dot, params, theta_x, theta_z):
    """
    Calculates and returns the linear velocity of each link of the snake.
    
    Args:
    - theta_x_dot: Angular velocity around the x-axis for each link.
    - theta_z_dot: Angular velocity around the z-axis for each link.
    - p_CM_dot: Velocity of the Center of Mass (CM) of the snake robot.
    - params: Dictionary containing necessary parameters and global variables.
    
    Returns:
    - XYZ_dot: Linear velocity of each link in the global coordinate frame.
    """
    # Extract necessary parameters from `params` dictionary
    e = params['e']
    l = params['l']
    K = params['K']

    cos_theta_z_diag = calculate_cos_theta_diag(theta_z)
    sin_theta_x_diag = calculate_sin_theta_diag(theta_x)
    cos_theta_x_diag = calculate_cos_theta_diag(theta_x)
    sin_theta_z_diag = calculate_sin_theta_diag(theta_z)

    
    # Calculate linear velocities
    x_dot = l * K.T @ (sin_theta_z_diag @ cos_theta_x_diag @ theta_z_dot + cos_theta_z_diag @ sin_theta_x_diag @ theta_x_dot) + e * p_CM_dot[0]
    y_dot = -l * K.T @ (-sin_theta_z_diag @ sin_theta_x_diag @ theta_z_dot + cos_theta_z_diag @ cos_theta_x_diag @ theta_x_dot) + e * p_CM_dot[1]
    z_dot = l * K.T @ cos_theta_z_diag @ theta_z_dot 
    
    # Construct the return vector
    XYZ_dot = vertcat(x_dot, y_dot, z_dot)
    
    return XYZ_dot