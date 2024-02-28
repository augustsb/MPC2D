import numpy as np
import casadi as ca

def init_model_parameters():
    print("Initializing model parameters...")

    # Initialize parameters dictionary
    params = {}

    # Ground friction model
    params['friction_model'] = 'MORISON'

    # Gravitational constant
    params['g'] = 9.81


    #Original
    # Kinematic and dynamic parameters
    params['n'] = 10  # Number of links
    params['m'] = 0.6597  # Mass of each link
    params['l'] = 0.07  # Length of each link (2*l for full length)
    params['m_tot'] = params['m'] * params['n']  # Total mass
  

    l = params['l']
    n = params['n']

    # Current effect
    v_x = -0.01
    v_y = -0.005
    v_z = -0.005

    V_x = v_x * ca.DM.ones((n, 1))
    V_y = v_y * ca.DM.ones((n, 1))
    V_z = v_z * ca.DM.ones((n, 1))

    V_current = 0 * ca.vertcat(V_x, V_y, V_z)
    params['V_current'] = V_current

    # Parameters for elliptical joint
    b, a = 0.05, 0.03

    ro = 1000
    C_f, C_d_x, C_d_z = 0.03, 2, 2
    params['ct'] = 0.5 * ro * np.pi * C_f * ((a + b) / 2) * 2 * l
    params['cn'] = 0.5 * ro * C_d_x * 2 * b * 2 * l
    params['cb'] = 0.5 * ro * C_d_z * 2 * a * 2 * l

    # Parameters for added mass forces and drag torque
    params['lambda_x'] = 1 / 6 * ro * np.pi * C_f * (a + b) * l**3
    params['lambda_z'] = 1 / 3 * ro * C_d_z * 2 * a * l**3

    # Moment of inertia for each link
    params['Jx'] = (params['m'] * a**2) / 4 + (params['m'] * l**2) / 3
    params['Jz'] = (params['m'] * b**2) / 4 + (params['m'] * l**2) / 3

    # Addition matrix A
    A = np.zeros((n-1, n))
    for i in range(n-1):
        A[i, i] = 1
        A[i, i+1] = 1
    params['A'] = A

   # Difference matrix D
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = 1
        D[i, i+1] = -1
    params['D'] = D

   # Reachability matrix R
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i:] = 1
    params['R'] = R

    # Summation vector
    e = np.ones((n, 1))
    params['e'] = e

    # Constant matrices K and V used in the model
    D_inv = np.linalg.pinv(D @ D.T)  # Using Moore-Penrose inverse for stability
    params['K'] = A.T @ D_inv @ D
    params['V'] = A.T @ D_inv @ A

    # Initial values
    n = params['n']
    params['theta_x0'] = np.zeros(n)
    params['theta_x0_dot'] = np.zeros(n)
    params['theta_z0'] = np.zeros(n)
    params['theta_z0_dot'] = np.zeros(n)
    params['p_CM0'] = np.array([0 ,0, 0])
    params['p_CM0_dot'] = np.array([0, 0, 0])
    params['phix0'] = np.zeros(n-1)
    params['phiz0'] = np.zeros(n-1)
    params['phix0_dot'] = np.zeros(n-1)
    params['phiz0_dot'] = np.zeros(n-1)


    # Visualization parameters
    params['visualize_motion'] = True
    params['visualizeCMposition'] = True
    params['timestep'] = 0.01
    params['fignumber'] = 100

    # Simulation data storage
    params['store_sim_data'] = True
    params['sim_data_save_interval'] = 0.001

    # AVI file creation parameters
    params['save_to_AVI'] = True
    params['AVI_file_name'] = 'test.avi'
    params['AVI_fps'] = 30
    params['AVI_speedfactor'] = 0.6
    params['AVI_include_axes'] = True

    return params