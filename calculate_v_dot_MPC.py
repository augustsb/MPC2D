from casadi import SX, vertcat
import casadi as ca
import numpy as np

from calculate_XYZ_dot import calculate_XYZ_dot 
from calculate_XYZ import calculate_XYZ
from calculate_joint_angles import calculate_joint_angles
from calculate_u_lateral_undulation import calculate_u_lateral_undulation
from calculate_u_lateral_undulation_MPC import calculate_u_lateral_undulation_MPC
from calculate_actuator_torques_USRFinal import calculate_actuator_torques_USRFinal
from calculate_q_dot_dot_CM_USR import calculate_q_dot_dot_CM_USR
from calculate_f_drag import calculate_f_drag
from LOS_guidance_3D import LOS_guidance_3D

def calculate_v_dot_MPC(t, current_state, params, controller_params, waypoint_params, p_pathframe, dimension):
#def calculate_v_dot_MPC(t, current_state, optimal_alpha_h, optimal_omega_h, optimal_delta_h, params, controller_params):
    # Assume params contains all necessary global variables and parameters
    n = params['n']

    if dimension == '3D':
    # Extract states from the state vector
        theta_x = current_state[0:n]
        theta_z  = current_state[n:2*n]
        p_CM = current_state[2*n:2*n+3]
        theta_x_dot = current_state[2*n+3:3*n+3]   
        theta_z_dot = current_state[3*n+3:4*n+3]
        p_CM_dot = current_state[4*n+3:4*n+6] 
        y_int = current_state[4*n+6:4*n+7]
        z_int = current_state[4*n+7:4*n+8]

    if dimension == '2D':
        theta_x = current_state[0:n]
        p_CM = current_state[n:n+3]
        theta_x_dot = current_state[n+3:2*n+3]   
        p_CM_dot = current_state[2*n+3:2*n+6] 
        y_int = current_state[2*n+6:2*n+7]
        z_int = 0
        theta_z = np.zeros(n)
        theta_z_dot = np.zeros(n)




    # Calculates the position of the links
    XYZ = calculate_XYZ(p_CM,params,theta_x, theta_z)

    #Calculates the linear velocities of the links

    XY_dot = calculate_XYZ_dot(theta_x_dot, theta_z_dot, p_CM_dot, params, theta_x, theta_z)
    
    #Calculates translational friction forces

    fx,fy,fz = calculate_f_drag(XY_dot, params, theta_x, theta_z)

    #Calculates joint angles and their derivatives

    phi_x, phi_z, phi_x_dot, phi_z_dot = calculate_joint_angles(theta_x, theta_z, theta_x_dot, theta_z_dot, params)

    #LOS guidance for waypoint following

    phi_o_x_commanded, phi_o_z_commanded = LOS_guidance_3D(theta_x, theta_z, p_pathframe, y_int, z_int, params, controller_params, waypoint_params)
    
    #Calculates control input according to lateral undulation without any directional control

    u_x, u_z, phi_ref_x = calculate_u_lateral_undulation(t, phi_x, phi_z, phi_x_dot, phi_z_dot, theta_x, theta_z, p_CM, params, controller_params, phi_o_x_commanded, phi_o_z_commanded)
    #u_x, u_z = calculate_u_lateral_undulation_MPC(t, phi_x, phi_x_dot,  params, controller_params, optimal_alpha_h, optimal_delta_h, optimal_omega_h)


    du_x, du_z = calculate_actuator_torques_USRFinal(u_x, u_z, theta_x, theta_z, params)

    # Calculates the acceleration of theta and p where p is the position of the CM only USR

    tau_x = np.zeros(n)
    tau_z = np.zeros(n)

    
    theta_x_dot_dot, p_CM_dot_dot, theta_z_dot_dot, y_int_dot, z_int_dot  = calculate_q_dot_dot_CM_USR( theta_x_dot, theta_z_dot,  fx, fy, fz, du_x, du_z, params, theta_x, theta_z, controller_params, y_int, z_int, p_pathframe)
    #y_int_dot = 0


    if dimension == '3D':
        v_dot = vertcat(theta_x_dot, theta_z_dot, p_CM_dot, theta_x_dot_dot,  theta_z_dot_dot, p_CM_dot_dot, y_int_dot, z_int_dot)
    
    if dimension == '2D':
        v_dot = vertcat(theta_x_dot, p_CM_dot, theta_x_dot_dot, p_CM_dot_dot, y_int_dot)



    elementwise_product_x = u_x * phi_x_dot  # Vector: [u_1*phi_dot_1, ..., u_N*phi_dot_N]
    elementwise_product_z = u_z * phi_z_dot
    # Absolute value of the product vector
    abs_elementwise_product_x = ca.fabs(elementwise_product_x)  # Vector: [|u_1*phi_dot_1|, ..., |u_N*phi_dot_N|]
    abs_elementwise_product_z = ca.fabs(elementwise_product_z)  # Vector: [|u_1*phi_dot_1|, ..., |u_N*phi_dot_N|]

    # Sum of absolute values
    energy_consumption_x = ca.sum1(abs_elementwise_product_x)
    energy_consumption_z = ca.sum1(abs_elementwise_product_z)
    energy_consumption = energy_consumption_x + energy_consumption_z

    #return v_dot, phi_x, phi_x_dot
    return v_dot, energy_consumption, phi_ref_x