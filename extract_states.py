
def extract_states(v_res, n):
    """
    Extracts individual state variables from the combined state vector.

    :param v_res: The combined state vector from the simulation results.
    :param n: The number of joints/links in the snake robot.
    :return: Individual state variables (theta_x, p_CM, theta_x_dot, p_CM_dot, theta_z, theta_z_dot).
    """
    # Assuming the structure of v_res follows the order:
    # [theta_x; p_CM; theta_x_dot; p_CM_dot; theta_z; theta_z_dot, y_int, z_int]
    theta_x = v_res[0:n]                        # Correct
    p_CM = v_res[n:n+3]                         # Correct
    theta_x_dot = v_res[n+3:2*n+3]              # Adjusted: starts right after p_CM
    p_CM_dot = v_res[2*n+3:2*n+6]               # Adjusted: starts right after theta_x_dot
    y_int = v_res[2*n+6:2*n+7]




    return theta_x, p_CM, theta_x_dot, p_CM_dot, y_int