from casadi import Opti, sumsqr, sqrt, Function, dot, MX, sumsqr, gradient, solve, norm_2, fmax, fmin
import numpy as np
import matplotlib.pyplot as plt
import traceback
from obstacle_methods import calculate_min_dist_to_obstacle


def calculate_safe_margin(alpha_h, direct_distance):
    # Assuming alpha_h is given in radians and direct_distance is the closest approach distance to the obstacle
    safe_margin = direct_distance * np.tan(alpha_h / 2)
    return safe_margin


def object_function(X, N):
    
    f = 0
    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point
        segment_length = sumsqr(X[:, i+1] - X[:, i]) #Works for some reason
        #segment_length = norm_2(X[:, i+1] - X[:, i]) #Does not work...
        f += segment_length
        #f += sumsqr(X[:,N-1] - X[:,i])
    
    return f


def mpc_shortest_path(current_p, target, obstacles, params, controller_params,  initial_N,  k,  result_queue, P,  P_sol):


    N = initial_N
    alpha_h = controller_params['alpha_h']

    opti = Opti()  # Create an optimization problem
    X = opti.variable(3, N)  # Position variables

    opti.set_initial(X[:,0], P[:,0])
    opti.set_initial(X[:,N-1], target)


    if (P_sol is not None):
        for i in range(1, N-1):
            opti.set_initial(X[:,i], P_sol[:,i])
    
    else:
        for i in range(N):
            opti.set_initial(X[:,i], P[:,i])


    #opti.subject_to(X[:, 0] == P[0,:])
    opti.subject_to(X[:, 0] == current_p)
    opti.subject_to(X[:, N-1] == target)

    for i in range(N-1):  # Clearance constraints for waypoints
        A = X[:, i]
        B = X[:, i+1]
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            #opti.subject_to(sumsqr(X[:, i] - o_pos) > (o_rad + alpha_h/2)**2)
            #opti.subject_to(norm_2(X[:, i] - o_pos) > (o_rad + alpha_h))
            min_dist_to_obstacle = calculate_min_dist_to_obstacle(A, B, o_pos, o_rad)
            opti.subject_to(min_dist_to_obstacle > alpha_h)
            #if i < N-1:  # Clearance constraints for midpoints
                #midpoint = (X[:, i] + X[:, i+1]) / 2
                #opti.subject_to(sumsqr(midpoint - o_pos) > (o_rad + alpha_h/2)**2)
                #opti.subject_to(norm_2(midpoint - o_pos) > (o_rad + alpha_h))


    f = object_function(X, N)
    opti.minimize(f)
    

  

    opts = {"verbose": True, "ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-1, "ipopt.constr_viol_tol": 1e-1, "expand": True, "ipopt.sb" : "yes"}  
    opti.solver('ipopt', opts)


    """
 
    opts = {
        "qpsol": "qrqp",  # Specify qrqp as the QP solver
        "qpsol_options": {  # Options for the qrqp solver
            #"constr_viol_tol": 1e-2,  # Constraint violation tolerance
            #"dual_inf_tol": 1e-2,  # Tolerance for convergence
            "error_on_fail": False,  # Allow the solver to continue even if it encounters an error
            # Note: max_iter option for qpsol isn't directly exposed here, it's a part of sqpmethod options
        },
        "expand": True,  # Enable problem expansion
        "verbose": True,  # sqpmethod verbosity
        "print_time": True,  # Print solver time
        # Use tol_pr and tol_du for specifying tolerance related to primal and dual infeasibilities
        #"tol_pr": 1e-5,  # Stopping criterion for primal infeasibility
        #"tol_du": 1e-5,  # Stopping criterion for dual infeasibility
        "max_iter": 1000,  # Maximum number of SQP iterations
    }

    opti.solver("sqpmethod", opts)


    """
    
    try:
        sol = opti.solve()
        solver_stats = opti.stats()
        solver_time = solver_stats.get('t_proc_total', None)
        sol_waypoints = sol.value(X)  # Extract the optimized waypoints if solution is feasible
        result_queue.put((sol_waypoints, solver_time))
        return

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
        solver_stats = opti.stats()
        solver_time = solver_stats.get('t_proc_total', None)
        #Do something to get rid of error
        if P_sol is not None:
            sol_waypoints = P_sol
        else:
            sol_waypoints = P[:N,:]

        result_queue.put((sol_waypoints, None))


        

            

    

    



