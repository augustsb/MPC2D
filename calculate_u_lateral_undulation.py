from casadi import SX, MX, sin, cos, sum1, pi
import numpy as np

def calculate_u_lateral_undulation(t, phi_x, phi_z, phi_x_dot, phi_z_dot, theta_x, theta_z, p_CM, params, controller_params, phi_o_x_commanded, phi_o_z_commanded):
#def calculate_u_lateral_undulation(t, phi_x, phi_x_dot,  params, controller_params, phi_ref_x, phi_ref_d_x):
    """
    Calculates and returns the actuator forces for the snake robot's joint controller.
    
    Args:
    - t: Current time.
    - phi_x, phi_z: Current joint angles.
    - phi_x_dot, phi_z_dot: Current joint angular velocities.
    - theta_x, theta_z: Current link angles.
    - p_CM: Position of the Center of Mass (CM) of the snake robot.
    - params: Dictionary containing necessary parameters and global variables.
    
    Returns:
    - u_x, u_z: Actuator forces for lateral undulation control.
    - phi_ref_x, phi_ref_z: Reference joint angles.
    """


    n = params['n']


    Kp_joint = controller_params['Kp_joint']
    Kd_joint = controller_params['Kd_joint']
    alpha_h = controller_params['alpha_h']
    omega_h = controller_params['omega_h']
    delta_h = controller_params['delta_h']
    alpha_v = controller_params['alpha_v']
    omega_v = controller_params['omega_v']
    delta_v = controller_params['delta_v']


    transition_in_progress = controller_params['transition_in_progress']
    if transition_in_progress:
        coeffs_list = controller_params['coeffs_list']
        print(coeffs_list)
        elapsed_time = t - controller_params['transition_start_time']
        print(elapsed_time)


    # Calculate references for joint angles
    phi_ref_x = MX.zeros(n-1)
    phi_ref_z = MX.zeros(n-1)
    phi_ref_d_x = MX.zeros(n-1)
    phi_ref_d_z = MX.zeros(n-1)
    phi_ref_dd_x = MX.zeros(n-1)
    phi_ref_dd_z = MX.zeros(n-1)
    u_x = MX.zeros(n-1)
    u_z = MX.zeros(n-1)

    for i in range(n-1):

        if (not transition_in_progress):


            phi_ref_x[i] = alpha_h * sin(omega_h * t + i * delta_h) - phi_o_x_commanded
            phi_ref_z[i] = alpha_v * sin(omega_v * t + i * delta_v) - phi_o_z_commanded

            phi_ref_d_x[i] = alpha_h * omega_h * cos(omega_h * t + i * delta_h)
            phi_ref_d_z[i] = alpha_v * omega_v * cos(omega_v * t + i * delta_v)

            phi_ref_dd_x[i] = -alpha_h * omega_h**2 * sin(omega_h * t + i * delta_h)
            phi_ref_dd_z[i] = -alpha_v * omega_v**2 * sin(omega_v * t + i * delta_v)

            error_x = phi_ref_x[i] - phi_x[i]
            error_x_d = phi_ref_d_x[i] - phi_x_dot[i]

            error_z = phi_ref_z[i] - phi_z[i]
            error_z_d = phi_ref_d_z[i] - phi_z_dot[i]

            u_x[i] = phi_ref_dd_x[i] - Kd_joint * error_x_d - Kp_joint * error_x
            u_z[i] = phi_ref_dd_z[i] - Kd_joint * error_z_d - Kp_joint * error_z

        else:

            coeffs = coeffs_list[i]
            current_pos = coeffs[0] + coeffs[1] * elapsed_time + coeffs[2] * elapsed_time**2 + coeffs[3] * elapsed_time**3 + coeffs[4] * elapsed_time**4 + coeffs[5] * elapsed_time**5
            current_vel = coeffs[1] + 2 * coeffs[2] * elapsed_time + 3 * coeffs[3] * elapsed_time**2 + 4 * coeffs[4] * elapsed_time**3 + 5 * coeffs[5] * elapsed_time**4
            current_acc = 2 * coeffs[2] + 6 * coeffs[3] * elapsed_time + 12 * coeffs[4] * elapsed_time**2 + 20 * coeffs[5] * elapsed_time**3


            phi_ref_x[i] = current_pos - phi_o_x_commanded
            phi_ref_d_x[i] = current_vel
            phi_ref_dd_x[i] = current_acc

            error_x = phi_ref_x[i] - phi_x[i]
            error_x_d = phi_ref_d_x[i] - phi_x_dot[i]

            u_x[i] =  current_acc - Kd_joint * error_x_d - Kp_joint * error_x
            u_z[i] = 0


    return u_x, u_z