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

def calculate_v_dot_MPC(t, current_state, params, controller_params, waypoint_params, p_pathframe):
#def calculate_v_dot_MPC(t, current_state, optimal_alpha_h, optimal_omega_h, optimal_delta_h, params, controller_params):
    # Assume params contains all necessary global variables and parameters
    n = params['n']
    # Extract states from the state vector
    theta_x = current_state[0:n]                        # Correct
    p_CM = current_state[n:n+3]                         # Correct
    theta_x_dot = current_state[n+3:2*n+3]              # Adjusted: starts right after p_CM
    p_CM_dot = current_state[2*n+3:2*n+6]               # Adjusted: starts right after theta_x_dot
    y_int = current_state[2*n+6:2*n+7]

    theta_z = np.zeros(n)
    theta_z_dot = np.zeros(n)
    z_int = 0
    #y_int = 0
    #p_pathframe = np.zeros(3)




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

    u_x, u_z = calculate_u_lateral_undulation(t, phi_x, phi_z, phi_x_dot, phi_z_dot, theta_x, theta_z, p_CM, params, controller_params, phi_o_x_commanded, phi_o_z_commanded)
    #u_x, u_z = calculate_u_lateral_undulation_MPC(t, phi_x, phi_x_dot,  params, controller_params, optimal_alpha_h, optimal_delta_h, optimal_omega_h)

    #Calculate the actuator torques

    #u_x = u # Assuming (n-1 ) control inputs for u_x
    u_z = np.zeros(n-1)  # Assuming (n-1) control inputs for u_z

    #u_z = np.zeros(n-1)

    du_x, du_z = calculate_actuator_torques_USRFinal(u_x, u_z, theta_x, theta_z, params)

    # Calculates the acceleration of theta and p where p is the position of the CM only USR

    tau_x = np.zeros(n)
    tau_z = np.zeros(n)

    
    theta_x_dot_dot, p_CM_dot_dot, theta_z_dot_dot, y_int_dot, z_int_dot  = calculate_q_dot_dot_CM_USR( theta_x_dot, theta_z_dot,  fx, fy, fz, du_x, du_z, params, theta_x, theta_z, controller_params, y_int, z_int, p_pathframe)
    #y_int_dot = 0

    # Calculating v_dot similar to MATLAB code, but using CasADi and Python syntax
    v_dot = vertcat(theta_x_dot, p_CM_dot, theta_x_dot_dot, p_CM_dot_dot, y_int_dot)
    #v_dot = vertcat(theta_x_dot, p_CM_dot, theta_x_dot_dot, p_CM_dot_dot)

    vec = ca.dot(u_x, phi_x_dot)
    #vec = ca.dot(du_x, phi_x_dot)
    energy_consumption = ca.sum1(vec)

    #return v_dot, phi_x, phi_x_dot
    return v_dot, energy_consumption