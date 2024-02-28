from casadi import SX, MX, DM, vertcat, mtimes, horzcat, diag, cos, sin
from calculate_diagonal_matrices import calculate_cos_theta_diag, calculate_sin_theta_diag
import casadi as ca

def calculate_f_drag(XYZ_dot, params, theta_x, theta_z):
    """
    Calculates and returns the translational friction forces on each link.
    
    Args:
    - XYZ_dot: The linear velocities of each link in the global coordinate frame.
    - params: Dictionary containing necessary parameters and global variables.
    
    Returns:
    - fx, fy, fz: Translational friction forces on each link in the x, y, z directions.
    """
    # Extract necessary parameters from `params` dictionary
    n = params['n']
    ct = params['ct']
    cn = params['cn']
    cb = params['cb']
    V_current = params['V_current']
    cos_theta_z_diag = calculate_cos_theta_diag(theta_z)
    sin_theta_x_diag = calculate_sin_theta_diag(theta_x)
    cos_theta_x_diag = calculate_cos_theta_diag(theta_x)
    sin_theta_z_diag = calculate_sin_theta_diag(theta_z)

    
    # The fluid forced with ZYX convention
    matrix1 = vertcat(
        horzcat(ct * MX.eye(n), MX.zeros(n, n), MX.zeros(n, n)),
        horzcat(MX.zeros(n, n), cn * MX.eye(n), MX.zeros(n, n)),
        horzcat(MX.zeros(n, n), MX.zeros(n, n), cb * MX.eye(n))
    )

    R_link_global = vertcat(
        horzcat(cos_theta_x_diag * cos_theta_z_diag, -sin_theta_x_diag, cos_theta_x_diag * sin_theta_z_diag),
        horzcat(sin_theta_x_diag * cos_theta_z_diag, cos_theta_x_diag, sin_theta_x_diag * sin_theta_z_diag),
        horzcat(-sin_theta_z_diag, MX.zeros(n,n), cos_theta_z_diag)
    )
    

    R_transpose = R_link_global.T

    matrix2 = mtimes(R_link_global, matrix1)
    matrix3 = mtimes(matrix2, R_transpose)

    f_linear = mtimes(matrix3, (XYZ_dot - V_current))

    matrix4 = ca.fabs(XYZ_dot - V_current) * (XYZ_dot - V_current)
    f_nonlinear = mtimes(matrix3, matrix4)
    f = -(f_linear + 0 * f_nonlinear)

    # Partition the vector
    fx = f[0:n]
    fy = f[n:2*n]
    fz = f[2*n:3*n]

    return fx, fy, fz