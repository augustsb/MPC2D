from casadi import SX, MX, sin, cos, sum1, pi
import numpy as np

#def calculate_u_lateral_undulation(t, phi_x, phi_z, phi_x_dot, phi_z_dot, theta_x, theta_z, p_CM, params, controller_params, phi_o_x_commanded, phi_o_z_commanded):
def calculate_u_lateral_undulation_MPC(t, phi_x, phi_x_dot,  params, controller_params, optimal_alpha_h, optimal_delta_h, optimal_omega_h):
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
    l = params['l']

    Kp_joint = controller_params['Kp_joint']
    Kd_joint = controller_params['Kd_joint']
    alpha_h = optimal_alpha_h
    omega_h = optimal_omega_h
    delta_h = optimal_delta_h
    alpha_v = controller_params['alpha_v']
    omega_v = controller_params['omega_v']
    delta_v = controller_params['delta_v']

 
    # Calculate references for joint angles
    phi_ref_x = MX.zeros(n-1)
    phi_ref_z = MX.zeros(n-1)
    for i in range(n-1):
        #phi_ref_x[i] = alpha_h * sin(omega_h * t + i * delta_h) - phi_offset_h
        phi_ref_x[i] = alpha_h * sin(omega_h * t + i * delta_h)
        #phi_ref_z[i] = alpha_v * sin(omega_v * t + i * delta_v + temp) - phi_offset_v
        phi_ref_z[i] = alpha_v * sin(omega_v * t + i * delta_v) 

    # Calculate references for joint velocities
    phi_ref_d_x = MX.zeros(n-1)
    phi_ref_d_z = MX.zeros(n-1)
    for i in range(n-1):
        phi_ref_d_x[i] = alpha_h * omega_h * cos(omega_h * t + i * delta_h)
        phi_ref_d_z[i] = alpha_v * omega_v * cos(omega_v * t + i * delta_v)

    # Calculate references for joint accelerations
    phi_ref_dd_x = MX.zeros(n-1)
    phi_ref_dd_z = MX.zeros(n-1)
    for i in range(n-1):
        phi_ref_dd_x[i] = -alpha_h * omega_h**2 * sin(omega_h * t + i * delta_h)
        phi_ref_dd_z[i] = -alpha_v * omega_v**2 * sin(omega_v * t + i * delta_v)


    # Calculate the actuator forces
    u_x = MX.zeros(n-1)
    u_z = MX.zeros(n-1)
    for i in range(n-1):
        error_x = phi_ref_x[i] - phi_x[i]
        error_x_d = phi_ref_d_x[i] - phi_x_dot[i]
        #u_x[i] = phi_ref_dd_x[i] - Kd_joint * error_x_d - Kp_joint * error_x
        u_x[i] = - Kd_joint * error_x_d - Kp_joint * error_x

        #error_z = phi_ref_z[i] - phi_z[i]
        #error_z_d = phi_ref_d_z[i] - phi_z_dot[i]
        #u_z[i] = phi_ref_dd_z[i] - Kd_joint * error_z_d - Kp_joint * error_z
        u_z[i] = 0


    return u_x, u_z