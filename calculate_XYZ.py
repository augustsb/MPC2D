from casadi import SX, DM, vertcat
from calculate_diagonal_matrices import calculate_cos_theta_diag, calculate_sin_theta_diag

# Assuming e, l, K, sin_theta_x_diag, cos_theta_x_diag, sin_theta_z_diag, cos_theta_z_diag are defined globally or passed as parameters

def calculate_XYZ(p_CM, params, theta_x, theta_z):
    """
    Calculates and returns the position of each link of the snake.
    
    Args:
    - p_CM: Position of the Center of Mass (CM) of the snake robot.
    - params: Dictionary containing necessary parameters and global variables.
    
    Returns:
    - XYZ: Position of each link in the global coordinate frame.
    """
    # Extract necessary parameters from `params` dictionary
    e = params['e']
    l = params['l']
    K = params['K']

    cos_theta_z_diag = calculate_cos_theta_diag(theta_z)
    sin_theta_x_diag = calculate_sin_theta_diag(theta_x)
    cos_theta_x_diag = calculate_cos_theta_diag(theta_x)
    sin_theta_z_diag = calculate_sin_theta_diag(theta_z)

    # Perform the calculations
    sin_theta_x = sin_theta_x_diag @ e
    cos_theta_x = cos_theta_x_diag @ e
    sin_theta_z = sin_theta_z_diag @ e
    
    # Calculate positions
    x = -l * K.T @ cos_theta_z_diag @ cos_theta_x + e * p_CM[0]
    y = -l * K.T @ cos_theta_z_diag @ sin_theta_x + e * p_CM[1]
    z = l * K.T @ sin_theta_z + e * p_CM[2]
  


    # Construct the return vector
    #XYZ = vertcat(x, y, z)
    XYZ = vertcat(x, y, z)
    
    return XYZ