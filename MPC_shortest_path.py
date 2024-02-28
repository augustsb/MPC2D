from casadi import Opti, sumsqr, sqrt, Function
import numpy as np
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from waypoint_methods import redistribute_waypoints, calculate_clearance


# wp α [deg] ω [deg/s] δ [deg] υ¯ [m/s] Pavg [W]

pareto_points = np.array([
    [0,     44.0100, 210.0000, 15.1400, 0.8425, 34.2515],
    [0.0500, 41.5451, 209.9863, 16.1863, 0.8407, 29.9318],
    [0.1000, 38.5042, 209.9987, 17.6406, 0.8327, 24.9031],
    [0.1500, 37.1659, 209.9976, 18.9259, 0.8259, 22.3614],
    [0.2000, 34.9161, 209.6835, 20.3226, 0.8107, 18.9214],
    [0.2500, 33.4248, 209.9899, 22.9651, 0.7922, 15.6914],
    [0.3000, 32.1065, 209.9997, 26.1707, 0.7640, 12.4822],
    [0.3500, 31.7042, 209.9932, 28.3385, 0.7444, 10.8197],
    [0.4000, 31.9861, 209.9938, 30.8753, 0.7233, 9.3646],
    [0.4500, 32.0642, 209.9931, 33.6360, 0.6956, 7.8220],
    [0.5000, 32.4826, 209.9995, 35.8916, 0.6731, 6.7981],
    [0.5500, 33.2925, 209.9890, 38.9056, 0.6417, 5.6396],
    [0.6000, 34.4218, 209.9967, 41.3291, 0.6170, 4.8916],
    [0.6500, 35.0257, 209.9667, 44.0756, 0.5842, 4.0883],
    [0.7000, 37.7160, 209.9899, 47.2997, 0.5523, 3.4617],
    [0.7500, 39.7087, 207.8852, 50.9907, 0.5060, 2.7433],
    [0.8000, 39.1360, 193.1456, 53.9248, 0.4379, 1.9467],
    [0.8500, 54.4381, 209.6399, 75.3818, 0.3102, 0.8571],
    [0.9000, 68.4280, 207.9591, 89.1905, 0.2425, 0.4111],
    [0.9500, 46.4469, 138.4423, 89.9360, 0.1555, 0.1693],
    [1.0000, 0, 16.5134, 59.8411, 0, 0]
])






def mpc_shortest_path(current_p,  target, obstacles, params, controller_params, initial_path, initial_N, k, result_queue):

    N = initial_N  # Starting with initial N
    safe_margin =  0.1
    opti = Opti()  # Create an optimization problem

    X = opti.variable(3, N+1)  # Position variables
    total_distance_objective = 0

    for h in range(N-1):  # Iterate over the horizon, except the last point where there's no next point
        segment_length = sumsqr(X[:, h+1] - X[:, h])
        total_distance_objective += segment_length
        total_distance_objective +=  sumsqr(target - X[:, h])

    # Initial position constraint
    opti.subject_to(X[:, 0] == current_p)
    opti.subject_to(X[:, N] == target)

    # Movement constraint between steps (simplified dynamics)
    for h in range(N-1):
        opti.subject_to(sumsqr(X[:, h+1] - X[:, h]) <= k)

    for h in range(N):  # Clearance constraints for waypoints
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            opti.subject_to(sumsqr(X[:, h] - o_pos) >= (o_rad + safe_margin)**2)
            if h < N-1:  # Clearance constraints for midpoints
                midpoint = (X[:, h] + X[:, h+1]) / 2
                opti.subject_to(sumsqr(midpoint - o_pos) >= (o_rad + safe_margin)**2)

    # Minimize the objective
    opti.minimize(total_distance_objective)

    # Solver options
    opts = {"verbose": True, "ipopt.print_level": 0, "ipopt.max_iter": 30, "ipopt.tol": 1e-3, "expand": True}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        sol_waypoints = sol.value(X).T  # Extract the optimized waypoints if solution is feasible
        distributed_waypoints = redistribute_waypoints(sol_waypoints, k, current_p)  # Redistribute waypoints
        #result_queue.put(distributed_waypoints)
        print("SOL:", sol_waypoints)
        print("dist:", distributed_waypoints)
        result_queue.put(sol_waypoints)
    except:
        print("FAILED!!")

    



