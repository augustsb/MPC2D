from casadi import SX, vertcat
import casadi as ca
import numpy as np

from calculate_XYZ_dot import calculate_XYZ_dot 
from calculate_XYZ import calculate_XYZ
from calculate_joint_angles import calculate_joint_angles
from calculate_u_lateral_undulation import calculate_u_lateral_undulation
from calculate_actuator_torques_USRFinal import calculate_actuator_torques_USRFinal
from calculate_q_dot_dot_CM_USR import calculate_q_dot_dot_CM_USR
from calculate_f_drag import calculate_f_drag
from LOS_guidance_3D import LOS_guidance_3D

#def calculate_v_dot(t, v, params, controller_params, waypoint_params, p_pathframe):
def calculate_v_dot(t, v, params, controller_params, waypoint_params, p_pathframe):

    # Assume params contains all necessary global variables and parameters
    #n = params['n']
    n = 10
    # Extract states from the state vector
    theta_x = v[0:n]                        # Correct
    p_CM = v[n:n+3]                         # Correct
    theta_x_dot = v[n+3:2*n+3]              # Adjusted: starts right after p_CM
    p_CM_dot = v[2*n+3:2*n+6]               # Adjusted: starts right after theta_x_dot
    y_int = v[2*n+6: 2*n+7]

    theta_z = np.zeros(n)
    theta_z_dot = np.zeros(n)
    z_int = 0
    p_pathframe = [0,0,0]


    # Calculates the position of the links
    XYZ = calculate_XYZ(p_CM,params,theta_x, theta_z)

    #Calculates the linear velocities of the links

    XYZ_dot = calculate_XYZ_dot(theta_x_dot, theta_z_dot, p_CM_dot, params, theta_x, theta_z)
    
    #Calculates translational friction forces

    fx,fy,fz = calculate_f_drag(XYZ_dot, params, theta_x, theta_z)

    #Calculates joint angles and their derivatives

    phi_x, phi_z, phi_x_dot, phi_z_dot = calculate_joint_angles(theta_x, theta_z, theta_x_dot, theta_z_dot, params)

    #LOS guidance for waypoint following

    phi_o_x_commanded, phi_o_z_commanded = LOS_guidance_3D(theta_x, theta_z, p_pathframe, y_int, z_int,  params, controller_params, waypoint_params)
    
    #Calculates control input according to lateral undulation without any directional control

    #u_x, u_z, phi_ref_x = calculate_u_lateral_undulation(t, phi_x, phi_x_dot,  params, controller_params, phi_o_x_commanded)
    u_x, u_z, phi_ref_x = calculate_u_lateral_undulation(t, phi_x, phi_z, phi_x_dot, phi_z_dot, theta_x, theta_z, p_CM, params, controller_params, phi_o_x_commanded, phi_o_z_commanded)

    #Calculate the actuator torques

    du_x, du_z = calculate_actuator_torques_USRFinal(u_x, u_z, theta_x, theta_z, params)

    # Calculates the acceleration of theta and p where p is the position of the CM only USR

    tau_x = np.zeros(n)
    tau_z = np.zeros(n)
    
    theta_x_dot_dot, p_CM_dot_dot, theta_z_dot_dot, y_int_dot, z_int_dot = calculate_q_dot_dot_CM_USR(theta_x_dot, theta_z_dot, fx, fy, fz, du_x, du_z, params, theta_x, theta_z,  controller_params, y_int, z_int, p_pathframe)


    # Calculating v_dot similar to MATLAB code, but using CasADi and Python syntax
    v_dot = vertcat(theta_x_dot, p_CM_dot, theta_x_dot_dot, p_CM_dot_dot, y_int_dot)

    return v_dot


    


