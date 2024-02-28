from casadi import SX, DM, MX, vertcat, horzcat, mtimes, diag, cos, sin, solve, sum1
import numpy as np
from calculate_diagonal_matrices import calculate_cos_theta_diag, calculate_sin_theta_diag

"""
def calculate_q_dot_dot_CM_USR( theta_x_dot,  fx, fy, fz, du_x, du_z, params, theta_x,  controller_params, y_int, p_pathframe):
    n = params['n']
    l = params['l']
    m = params['m']
    m_tot = params['m_tot']
    Jx = params['Jx']
    Jz = params['Jz']
    A = params['A']
    D = params['D']
    K = params['K']
    e = params['e']
    lambda_x = params['lambda_x']
    lambda_z = params['lambda_z']
    
    cos_theta_z_diag = np.eye(n,n)
    sin_theta_x_diag = calculate_sin_theta_diag(theta_x)
    cos_theta_x_diag = calculate_cos_theta_diag(theta_x)
    sin_theta_z_diag = np.zeros((n,n))
    theta_z_dot = np.zeros(n)

    V = params['V']

    # Define matrices
    Mtot_inv = MX.eye(3) / m_tot
    M11, M22, M33 = Mtot_inv[0, 0], Mtot_inv[1, 1], Mtot_inv[2, 2]

    # Calculation for M_theta
    M11_theta = Jx * cos_theta_z_diag**2 + l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag + \
                l**2 * m * cos_theta_z_diag * cos(sin_theta_x_diag) * V * cos_theta_z_diag * cos(sin_theta_x_diag)
                
    
    M12_theta = l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag - \
    l**2* m * cos_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag

    M21_theta = l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag - \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag

    M22_theta = Jz + l**2 * m * cos_theta_z_diag * V * cos_theta_z_diag + \
    l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag

    #Calculation for W_theta
    W11_theta = l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag - \
    l**2 * m * cos_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    W12_theta = W11_theta

    W21_theta = l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    W22_theta = -l**2 * m * cos_theta_z_diag * V * sin_theta_z_diag + \
    l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    #Calculation for V_theta
    V11_theta = -2 * l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag - \
    2 * l**2 * m * cos_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag

    V22_theta = -2 * l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag + \
    2 * l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag

    #Calculation for N_theta

    N11_theta = - l * sin_theta_x_diag * K
    N12_theta = l * cos_theta_x_diag * K
    N13_theta = np.zeros((n,n))
    N21_theta = - l * sin_theta_z_diag * cos_theta_x_diag * K
    N22_theta = - l * sin_theta_z_diag * sin_theta_x_diag * K
    N23_theta = - l * cos_theta_z_diag * K

    M_theta_top = horzcat(M11_theta, M12_theta)  # Combine horizontally
    M_theta_bottom = horzcat(M21_theta, M22_theta)
    M_theta = vertcat(M_theta_top, M_theta_bottom)  # Then combine vertically

    # For W_theta
    W_theta_top = horzcat(W11_theta, W12_theta)
    W_theta_bottom = horzcat(W21_theta, W22_theta)
    W_theta = vertcat(W_theta_top, W_theta_bottom)

    # For V_theta, assuming you want to combine V11_theta with zeros and V22_theta with zeros
    V_theta_top = horzcat(V11_theta, MX.zeros(n, n))
    V_theta_bottom = horzcat(MX.zeros(n, n), V22_theta)
    V_theta = vertcat(V_theta_top, V_theta_bottom)

    # For N_theta, assuming N*_theta are vectors or matrices that need to be combined in a specific manner
    N_theta_top = horzcat(N11_theta, N12_theta, N13_theta)
    N_theta_bottom = horzcat(N21_theta, N22_theta, N23_theta)
    N_theta = vertcat(N_theta_top, N_theta_bottom)


    # CasADi symbolic identity and zero matrices
    I_casadi = MX.eye(n)
    Z_casadi = MX.zeros(n, n)

    # Directly construct the block matrix in CasADi
    P = vertcat(horzcat(Z_casadi, I_casadi), horzcat(I_casadi, Z_casadi))


    I_tau = vertcat(
        vertcat(-np.eye(n), np.zeros((n,n))).T,
        vertcat(np.zeros((n,n)),  -np.eye(n)).T
    )


    # Calculating theta_dot_sq and theta_dot_abs if necessary
    #theta_dot_sq = theta_x_dot**2 + theta_z_dot**2


    theta_x_dot_sq = theta_x_dot * theta_x_dot
    theta_z_dot_sq = theta_z_dot * theta_z_dot
    theta_dot = vertcat(theta_x_dot, theta_z_dot)
    theta_dot_sq = vertcat(theta_x_dot_sq, theta_z_dot_sq)

    
    I = MX.eye(2*n)  # Create an identity matrix of size 2*n
    # Solve the system M_theta * M_inv = I for M_inv
    M_inv = solve(M_theta, I)

    # Ensure taf_x is a vector. For example:
    taf_x = -lambda_x * mtimes(cos_theta_z_diag**2, theta_x_dot)  # Element-wise multiplication if cos_theta_z_diag is a vector


    # Ensure taf_z is also a vector
    taf_z = -lambda_z * theta_z_dot

    # Now, assuming both taf_x and taf_z are vectors of compatible dimensions, concatenate them
    tau_fluid = vertcat(taf_x, taf_z)

    f_fluid = vertcat(fx,fy,fz)

    u = vertcat(du_x, du_z)

    # Create an n x n identity matrix and then select the first n-1 columns to simulate an n x (n-1) identity matrix
    I_n_minus_1 = np.eye(n)[:, :n-1]
    Z_n_minus_1 = np.zeros((n, n-1))
    top_block = np.hstack((I_n_minus_1, Z_n_minus_1))
    bottom_block = np.hstack((Z_n_minus_1, I_n_minus_1))
    U_dt = np.vstack((top_block, bottom_block))


    # Calculating W_th
    P_theta_dot = mtimes(P, theta_dot)
    diag_theta_dot = diag(theta_dot)  # This creates a diagonal matrix from theta_dot
    W_th = W_theta*theta_dot_sq + V_theta*mtimes(diag_theta_dot, P_theta_dot) + mtimes(N_theta,f_fluid) - I_tau*tau_fluid

    # Calculate theta_dot_dot
    theta_dot_dot = mtimes(M_inv, (mtimes(U_dt, u) - W_th))

    theta_x_dot_dot = theta_dot_dot[:n]
    #theta_z_dot_dot = theta_dot_dot[n:2*n] 



    px_dot_dot = mtimes(M11, mtimes(e.T, fx))
    py_dot_dot = mtimes(M22, mtimes(e.T,fy))
    #pz_dot_dot = mtimes(M33, mtimes(e.T,fz))

    p_CM_dot_dot = vertcat(px_dot_dot, py_dot_dot)

    delta_y = controller_params['delta_y']
    sigma_y = controller_params['sigma_y']

    y_int_dot = (delta_y * p_pathframe[1]) / ((p_pathframe[1] + sigma_y * y_int)**2 + delta_y**2)

    return theta_x_dot_dot, p_CM_dot_dot, y_int_dot

"""

from casadi import SX, DM, MX, vertcat, horzcat, mtimes, diag, cos, sin, solve, sum1
import numpy as np
from calculate_diagonal_matrices import calculate_cos_theta_diag, calculate_sin_theta_diag

def calculate_q_dot_dot_CM_USR( theta_x_dot, theta_z_dot,  fx, fy, fz, du_x, du_z, params, theta_x, theta_z, controller_params, y_int, z_int, p_pathframe):
    n = params['n']
    l = params['l']
    m = params['m']
    m_tot = params['m_tot']
    Jx = params['Jx']
    Jz = params['Jz']
    A = params['A']
    D = params['D']
    K = params['K']
    e = params['e']
    lambda_x = params['lambda_x']
    lambda_z = params['lambda_z']
    
    cos_theta_z_diag = calculate_cos_theta_diag(theta_z)
    sin_theta_x_diag = calculate_sin_theta_diag(theta_x)
    cos_theta_x_diag = calculate_cos_theta_diag(theta_x)
    sin_theta_z_diag = calculate_sin_theta_diag(theta_z)

    V = params['V']

    # Define matrices
    Mtot_inv = MX.eye(3) / m_tot
    M11, M22, M33 = Mtot_inv[0, 0], Mtot_inv[1, 1], Mtot_inv[2, 2]

    # Calculation for M_theta
    M11_theta = Jx * cos_theta_z_diag**2 + l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag + \
                l**2 * m * cos_theta_z_diag * cos(sin_theta_x_diag) * V * cos_theta_z_diag * cos(sin_theta_x_diag)
    
    M12_theta = l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag - \
    l**2* m * cos_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag

    M21_theta = l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag - \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag

    M22_theta = Jz + l**2 * m * cos_theta_z_diag * V * cos_theta_z_diag + \
    l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag

    #Calculation for W_theta
    W11_theta = l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag - \
    l**2 * m * cos_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    W12_theta = W11_theta

    W21_theta = l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    W22_theta = -l**2 * m * cos_theta_z_diag * V * sin_theta_z_diag + \
    l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * cos_theta_z_diag * cos_theta_x_diag + \
    l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * cos_theta_z_diag * sin_theta_x_diag

    #Calculation for V_theta
    V11_theta = -2 * l**2 * m * cos_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag - \
    2 * l**2 * m * cos_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag

    V22_theta = -2 * l**2 * m * sin_theta_z_diag * cos_theta_x_diag * V * sin_theta_z_diag * sin_theta_x_diag + \
    2 * l**2 * m * sin_theta_z_diag * sin_theta_x_diag * V * sin_theta_z_diag * cos_theta_x_diag

    #Calculation for N_theta

    N11_theta = - l * sin_theta_x_diag * K
    N12_theta = l * cos_theta_x_diag * K
    N13_theta = np.zeros((n,n))
    N21_theta = - l * sin_theta_z_diag * cos_theta_x_diag * K
    N22_theta = - l * sin_theta_z_diag * sin_theta_x_diag * K
    N23_theta = - l * cos_theta_z_diag * K

    M_theta_top = horzcat(M11_theta, M12_theta)  # Combine horizontally
    M_theta_bottom = horzcat(M21_theta, M22_theta)
    M_theta = vertcat(M_theta_top, M_theta_bottom)  # Then combine vertically

    # For W_theta
    W_theta_top = horzcat(W11_theta, W12_theta)
    W_theta_bottom = horzcat(W21_theta, W22_theta)
    W_theta = vertcat(W_theta_top, W_theta_bottom)

    # For V_theta, assuming you want to combine V11_theta with zeros and V22_theta with zeros
    V_theta_top = horzcat(V11_theta, MX.zeros(n, n))
    V_theta_bottom = horzcat(MX.zeros(n, n), V22_theta)
    V_theta = vertcat(V_theta_top, V_theta_bottom)

    # For N_theta, assuming N*_theta are vectors or matrices that need to be combined in a specific manner
    N_theta_top = horzcat(N11_theta, N12_theta, N13_theta)
    N_theta_bottom = horzcat(N21_theta, N22_theta, N23_theta)
    N_theta = vertcat(N_theta_top, N_theta_bottom)


    # CasADi symbolic identity and zero matrices
    I_casadi = MX.eye(n)
    Z_casadi = MX.zeros(n, n)

    # Directly construct the block matrix in CasADi
    P = vertcat(horzcat(Z_casadi, I_casadi), horzcat(I_casadi, Z_casadi))


    I_tau = vertcat(
        vertcat(-np.eye(n), np.zeros((n,n))).T,
        vertcat(np.zeros((n,n)),  -np.eye(n)).T
    )


    # Calculating theta_dot_sq and theta_dot_abs if necessary
    #theta_dot_sq = theta_x_dot**2 + theta_z_dot**2


    theta_x_dot_sq = theta_x_dot * theta_x_dot
    theta_z_dot_sq = theta_z_dot * theta_z_dot
    theta_dot = vertcat(theta_x_dot, theta_z_dot)
    theta_dot_sq = vertcat(theta_x_dot_sq, theta_z_dot_sq)

    
    I = MX.eye(2*n)  # Create an identity matrix of size 2*n
    # Solve the system M_theta * M_inv = I for M_inv
    M_inv = solve(M_theta, I)

    # Ensure taf_x is a vector. For example:
    taf_x = -lambda_x * mtimes(cos_theta_z_diag**2, theta_x_dot)  # Element-wise multiplication if cos_theta_z_diag is a vector


    # Ensure taf_z is also a vector
    taf_z = -lambda_z * theta_z_dot

    # Now, assuming both taf_x and taf_z are vectors of compatible dimensions, concatenate them
    tau_fluid = vertcat(taf_x, taf_z)

    f_fluid = vertcat(fx,fy,fz)

    u = vertcat(du_x, du_z)

    # Create an n x n identity matrix and then select the first n-1 columns to simulate an n x (n-1) identity matrix
    I_n_minus_1 = np.eye(n)[:, :n-1]
    Z_n_minus_1 = np.zeros((n, n-1))
    top_block = np.hstack((I_n_minus_1, Z_n_minus_1))
    bottom_block = np.hstack((Z_n_minus_1, I_n_minus_1))
    U_dt = np.vstack((top_block, bottom_block))


    # Calculating W_th
    P_theta_dot = mtimes(P, theta_dot)
    diag_theta_dot = diag(theta_dot)  # This creates a diagonal matrix from theta_dot
    W_th = W_theta*theta_dot_sq + V_theta*mtimes(diag_theta_dot, P_theta_dot) + mtimes(N_theta,f_fluid) - I_tau*tau_fluid

    # Calculate theta_dot_dot
    theta_dot_dot = mtimes(M_inv, (mtimes(U_dt, u) - W_th))

    theta_x_dot_dot = theta_dot_dot[:n]
    theta_z_dot_dot = theta_dot_dot[n:2*n] 



    px_dot_dot = mtimes(M11, mtimes(e.T, fx))
    py_dot_dot = mtimes(M22, mtimes(e.T,fy))
    pz_dot_dot = mtimes(M33, mtimes(e.T,fz))

    p_CM_dot_dot = vertcat(px_dot_dot, py_dot_dot, pz_dot_dot)

    delta_z = controller_params['delta_z']
    delta_y = controller_params['delta_y']
    sigma_z = controller_params['sigma_z']
    sigma_y = controller_params['sigma_y']

    #z_int_dot = (delta_z * p_pathframe[2]) / ((p_pathframe[2] + sigma_z * z_int)**2 + delta_z**2)
    z_int_dot = 0
    y_int_dot = (delta_y * p_pathframe[1]) / ((p_pathframe[1] + sigma_y * y_int)**2 + delta_y**2)

    return theta_x_dot_dot, p_CM_dot_dot, theta_z_dot_dot, y_int_dot, z_int_dot