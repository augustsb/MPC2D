import numpy as np

def extract_states(v_res, n, dimension):
    """
    Extracts individual state variables from the combined state vector.

    :param v_res: The combined state vector from the simulation results.
    :param n: The number of joints/links in the snake robot.
    :return: Individual state variables (theta_x, p_CM, theta_x_dot, p_CM_dot, theta_z, theta_z_dot).
    """
    # Assuming the structure of v_res follows the order:
    # [theta_x; p_CM; theta_x_dot; p_CM_dot; theta_z; theta_z_dot, y_int, z_int]

    if dimension == '3D':
        theta_x = v_res[0:n]
        theta_z  = v_res[n:2*n]
        p_CM = v_res[2*n:2*n+3]
        theta_x_dot = v_res[2*n+3:3*n+3]   
        theta_z_dot = v_res[3*n+3:4*n+3]
        p_CM_dot = v_res[4*n+3:4*n+6] 
        y_int = v_res[4*n+6:4*n+7]
        z_int = v_res[4*n+7:4*n+8]

    if dimension == '2D':
        theta_x = v_res[0:n]
        p_CM = v_res[n:n+3]
        theta_x_dot = v_res[n+3:2*n+3]   
        p_CM_dot = v_res[2*n+3:2*n+6] 
        y_int = v_res[2*n+6:2*n+7]
        theta_z = np.zeros(n)
        theta_z_dot = np.zeros(n)
        z_int = 0


    return theta_x, theta_z, p_CM, theta_x_dot, theta_z_dot, p_CM_dot, y_int, z_int