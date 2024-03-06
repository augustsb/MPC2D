from casadi import Opti, sumsqr, sqrt, Function, dot, MX, sumsqr, gradient, solve
import osqp
import numpy as np
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from waypoint_methods import redistribute_waypoints, generate_initial_path, path_resolution, extend_horizon, calculate_total_distance
from euclid import *
import traceback



def mpc_shortest_path(current_p, target, obstacles, params, controller_params,  initial_N,  k,  result_queue, P):

    safe_margin = controller_params['alpha_h']

    total_distance_objective = 0


    N = min(initial_N, P.shape[0])

    opti = Opti()  # Create an optimization problem
    X = opti.variable(3, N)  # Position variables

    opti.subject_to(X[:, N-1] == P[N-1,:])
    opti.subject_to(sumsqr(X[:, 0] - P[0,:]) <= 0.1)

    if 'sol_waypoints' in locals() or 'sol_waypoints' in globals():
        opti.set_initial(X, P.T)

    for h in range(N-1):  # Iterate over the horizon, except the last point where there's no next point
        segment_length = sumsqr(X[:, h+1] - X[:, h])
        total_distance_objective += segment_length
        total_distance_objective +=   sumsqr(P[N-1,:] - X[:,h])

    # Movement constraint between steps. Slight constant for flexibility
    for h in range(N-2):
        opti.subject_to(sumsqr(X[:, h+1] - X[:, h]) <= (k + 0.1) **2)
        #opti.subject_to(sumsqr(X[:, h+1] - X[:, h]) >= (k - 0.5) **2)


    for h in range(N-1):  # Clearance constraints for waypoints
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            opti.subject_to(sumsqr(X[:, h] - o_pos) >= (o_rad + safe_margin)**2)
            if h < N-1:  # Clearance constraints for midpoints
                midpoint = (X[:, h] + X[:, h+1]) / 2
                opti.subject_to(sumsqr(midpoint - o_pos) >= (o_rad + safe_margin)**2)

    opti.minimize(total_distance_objective)
    opts = {"verbose": True, "ipopt.print_level": 1, "ipopt.max_iter": 50, "ipopt.tol": 1e-2, "ipopt.constr_viol_tol": 1e-2, "expand": True,}  
    opti.solver('ipopt', opts)
    
    try:
        sol = opti.solve()
        sol_waypoints = sol.value(X)  # Extract the optimized waypoints if solution is feasible
        result_queue.put(sol_waypoints)

    except Exception as e:
        traceback.print_exc()  # This will print the traceback of the exception
        #Debugging....
        print(f"Optimization failed with error: {e}")
        print("Optimization failed. Investigating variables...")
        print("Current (real) position:", opti.debug.value(P[0,:]))
        print("Target (real) position:", opti.debug.value(P[-1, :]))
        print("Current position:", opti.debug.value(X[:,0]))
        print("Target position:", opti.debug.value(X[:, N-1]))
        print("Infeasibilities:", opti.debug.show_infeasibilities())
        #P, num_states = generate_initial_path(current_p, P[N-1,:], k)
        sol_waypoints = P.T
        result_queue.put(sol_waypoints)
        #Do something to get rid of error
        

            

    

    



