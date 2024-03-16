import casadi as ca

def LOS_guidance_3D(theta_x, theta_z, p_pathframe, y_int, z_int, params, controller_params, waypoint_params):

    # Extract necessary parameters from the params dictionary


    n = params['n']
    delta_y = controller_params['delta_y']
    delta_z = controller_params['delta_z']
    sigma_y = controller_params['sigma_y']
    sigma_z = controller_params['sigma_z']
    cur_alpha_path = waypoint_params['cur_alpha_path']
    cur_gamma_path = waypoint_params['cur_gamma_path']
    K_theta = controller_params['K_theta']
    K_psi = controller_params['K_psi']
    
    # Estimates the direction of the snake wrt the path frame
    psi_bar_x = 1/n * ca.sum1(theta_x)
    psi_bar_z = 1/n * ca.sum1(theta_z)


    # Integral line of sight
    psi_ref_x = cur_alpha_path - ca.atan2(p_pathframe[1] + sigma_y * y_int, delta_y)
    psi_ref_z = cur_gamma_path + ca.atan2(p_pathframe[2] + sigma_z * z_int, delta_z)


    phi_o_x_commanded = K_theta*(psi_bar_x - psi_ref_x)
    phi_o_z_commanded = K_psi*(psi_bar_z - psi_ref_z)

    return phi_o_x_commanded, phi_o_z_commanded