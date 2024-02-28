import casadi as ca

def calculate_joint_angles(theta_x, theta_z, theta_x_dot, theta_z_dot, params):
    n = params['n']
    phi_x = ca.MX.zeros(n-1, 1)
    phi_z = ca.MX.zeros(n-1, 1)
    phi_x_dot = ca.MX.zeros(n-1, 1)
    phi_z_dot = ca.MX.zeros(n-1, 1)

    for i in range(n-1):  # Python indexing starts at 0
        R_link_global_T = ca.vertcat(
            ca.horzcat(ca.cos(theta_x[i]) * ca.cos(theta_z[i]), ca.sin(theta_x[i]) * ca.cos(theta_z[i]), -ca.sin(theta_z[i])),
            ca.horzcat(-ca.sin(theta_x[i]), ca.cos(theta_x[i]), 0),
            ca.horzcat(ca.cos(theta_x[i]) * ca.sin(theta_z[i]), ca.sin(theta_x[i]) * ca.sin(theta_z[i]), ca.cos(theta_z[i]))
        )

        R_link_global1 = ca.vertcat(
            ca.horzcat(ca.cos(theta_x[i+1]) * ca.cos(theta_z[i+1]), -ca.sin(theta_x[i+1]), ca.cos(theta_x[i+1]) * ca.sin(theta_z[i+1])),
            ca.horzcat(ca.sin(theta_x[i+1]) * ca.cos(theta_z[i+1]), ca.cos(theta_x[i+1]), ca.sin(theta_x[i+1]) * ca.sin(theta_z[i+1])),
            ca.horzcat(-ca.sin(theta_z[i+1]), 0, ca.cos(theta_z[i+1]))
        )

        R_i1_i = R_link_global_T @ R_link_global1
        phi_x[i] = -ca.atan2(R_i1_i[0, 1], R_i1_i[1, 1])
        phi_z[i] = ca.atan2(R_i1_i[0, 2], R_i1_i[0, 0])

        R_i1_i1 = R_i1_i.T
        vector_001 = ca.vertcat(0, 0, 1)
        vector_010 = ca.vertcat(0, 1, 0)
        phi_x_dot[i] = theta_x_dot[i+1] - ca.mtimes(vector_001.T, ca.mtimes(R_i1_i1, ca.vertcat(0, 0, theta_x_dot[i])))
        phi_z_dot[i] = theta_z_dot[i+1] - ca.mtimes(vector_010.T, ca.mtimes(R_i1_i1, ca.vertcat(0, theta_z_dot[i], 0)))

    return phi_x, phi_z, phi_x_dot, phi_z_dot