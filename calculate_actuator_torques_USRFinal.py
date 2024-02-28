import casadi as ca

def calculate_actuator_torques_USRFinal(u_x, u_z, theta_x, theta_z, params):
    # Initialize the output vectors
    du_x1 = []
    du_z1 = []
    du_x2 = []
    du_z2 = []

    u_x = ca.vertcat(0, u_x)
    u_z = ca.vertcat(0, u_z)

    n = params['n']

    for i in range(1, n):
        # Correct way to construct matrices in CasADi
        R_link_global_T = ca.vertcat(
            ca.horzcat(ca.cos(theta_x[i]) * ca.cos(theta_z[i]), ca.sin(theta_x[i]) * ca.cos(theta_z[i]), -ca.sin(theta_z[i])),
            ca.horzcat(-ca.sin(theta_x[i]), ca.cos(theta_x[i]), 0),
            ca.horzcat(ca.cos(theta_x[i]) * ca.sin(theta_z[i]), ca.sin(theta_x[i]) * ca.sin(theta_z[i]), ca.cos(theta_z[i]))
        )

        R_link_global1 = ca.vertcat(
            ca.horzcat(ca.cos(theta_x[i-1]) * ca.cos(theta_z[i-1]), -ca.sin(theta_x[i-1]), ca.cos(theta_x[i-1]) * ca.sin(theta_z[i-1])),
            ca.horzcat(ca.sin(theta_x[i-1]) * ca.cos(theta_z[i-1]), ca.cos(theta_x[i-1]), ca.sin(theta_x[i-1]) * ca.sin(theta_z[i-1])),
            ca.horzcat(-ca.sin(theta_z[i-1]), 0, ca.cos(theta_z[i-1]))
        )

        R_i1_i = R_link_global_T @ R_link_global1

        T_inv = ca.vertcat(
            ca.horzcat(1, 0, -ca.sin(theta_z[i-1])),
            ca.horzcat(0, 1, 0),
            ca.horzcat(0, 0, ca.cos(theta_z[i-1]))
        )

        du = ca.vertcat(0, 0, u_x[i]) - R_i1_i @ ca.vertcat(0, 0, u_x[i-1])
        du = T_inv.T @ du

        du_x1.append(du[2])
        du_z1.append(du[1])

    for i in range(n - 1):
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

        T_inv = ca.vertcat(
            ca.horzcat(1, 0, -ca.sin(theta_z[i])),
            ca.horzcat(0, 1, 0),
            ca.horzcat(0, 0, ca.cos(theta_z[i]))
        )

        du = R_i1_i @ ca.vertcat(0, u_z[i+1], 0) - ca.vertcat(0, u_z[i], 0)
        du = T_inv.T @ du

        du_x2.append(du[2])
        du_z2.append(du[1])

    du_x = ca.vertcat(*du_x1) + ca.vertcat(*du_x2)
    du_z = ca.vertcat(*du_z1) + ca.vertcat(*du_z2)

    return du_x, du_z


