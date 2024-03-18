from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs, norm_2
from obstacle_methods import calculate_min_dist_to_obstacle
import numpy as np
import matplotlib.pyplot as plt
import traceback



#data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)

def object_function_alpha(X, alpha_h, V, N):

    #Delta 20,30,40
    #intercept = MX([1.14860066])
    #coefficients  = MX([ 0.0, -7.58928297, -5.0892465,   7.37952851, 25.62503367,  1.19951258])

    #delta 20
    #intercept = MX([2.41289405])
    #coefficients = MX([0.0, -25.38440826, -3.36016491, 41.68196734, 28.05986638, 0.0832687 ])
    
    #delta40 max_energy8
    #intercept = MX([1.57621037])
    #coefficients = MX([ 0.0, -12.16753308,  -3.65471241,  10.85165704,  27.98929701, -1.68525643])

    #delta40 max_energy20
    intercept = MX([3.74085757])
    coefficients = MX([  0.0, -28.36466061, -4.36192943,  24.66493004,  45.71213631, -6.37794483])

    f_dist = MX(0)
    f_energy = MX(0)
    f_time = MX(0)
    epsilon = 1e-8

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        f_dist += segment_length

        alpha_h_i = alpha_h[i]
        V_i = V[i] + epsilon  # Add epsilon to avoid division by zero
        #V_i = V[i]
        # Correct construction of polynomial features for iteration i
        linear_terms = vertcat(alpha_h_i,  V_i)
        squared_terms = vertcat(alpha_h_i**2,  V_i**2)
        interaction_terms = vertcat(alpha_h_i*V_i)
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept

        """
        # Extend construction of polynomial features for iteration i to include 3rd-order terms
        linear_terms = vertcat(alpha_h_i, V_i)
        squared_terms = vertcat(alpha_h_i**2, V_i**2)
        cubic_terms = vertcat(alpha_h_i**3, V_i**3)
        interaction_terms_linear = alpha_h_i * V_i  # 1st order interaction
        interaction_terms_squared = alpha_h_i**2 * V_i + alpha_h_i * V_i**2  # 2nd order interactions
        interaction_terms_cubic = alpha_h_i**2 * V_i**2  # Example cubic interaction term (adjust as needed)

        # Combine all terms, adjusting the vector to match your model's expected input
        all_terms = vertcat(1, linear_terms, squared_terms, cubic_terms,
                            interaction_terms_linear, interaction_terms_squared, interaction_terms_cubic)
        
        """

        predicted_time = segment_length / V_i
        predicted_average_energy = intercept + dot(coefficients, all_terms)

        f_time += predicted_time
        f_energy += predicted_average_energy
 

    total_cost = 5*f_dist  + 5*f_energy + 100*f_time

    return total_cost

    #return f_energy

def object_function(X, N):

    f_dist = 0

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point
        segment_length = sumsqr(X[:, i+1] - X[:, i])
        #segment_length = norm_2(X[:, i+1] - X[:, i])
        f_dist += segment_length

    return f_dist


def mpc_energy_efficiency_alpha(current_p, p_dot, target, obstacles, params, controller_params, initial_N, k, result_queue, P, P_sol):

    
    alpha_h0 = controller_params['alpha_h']
    v0 = np.linalg.norm(p_dot)

    min_velocity = controller_params['v_min']
    max_velocity = controller_params['v_max']
    alpha_h_min = controller_params['alpha_h_min']
    alpha_h_max = controller_params['alpha_h_max']

    N = initial_N

    opti = Opti()  # Create an optimization problem

    X = opti.variable(3, N)  # Position variables
    V = opti.variable(N)
    alpha_h = opti.variable(N)

    
    opti.set_initial(X[:,0], P[:,0])
    opti.set_initial(X[:,N-1], target)
    opti.set_initial(alpha_h[0], alpha_h_max)

    if (P_sol is not None):
        for i in range(1, N-1):
            opti.set_initial(X[:,i], P_sol[:,i])
    
    else:
        for i in range(N):
            opti.set_initial(X[:,i], P[:,i])

  
    #opti.subject_to(alpha_h[0] == alpha_h0)
    opti.subject_to(X[:, 0] == current_p)
    opti.subject_to(X[:, N-1] == target)
    for i in range(N):
        opti.subject_to(opti.bounded(alpha_h_min, alpha_h[i], alpha_h_max))
        opti.subject_to(opti.bounded(min_velocity, V[i], max_velocity))

    #clearance_base = 0.1  # Base clearance
    #clearance_velocity_factor = 0.05  # Factor to scale clearance with velocity
    for i in range(N-1):  # Clearance constraints for waypoints
        A = X[:, i]
        B = X[:, i+1]
        alpha_h_i = alpha_h[i]
        #V_i = V[i]
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            min_dist_to_obstacle = calculate_min_dist_to_obstacle(A, B, o_pos, o_rad)
            #required_clearance = alpha_h_i + clearance_base + clearance_velocity_factor * V_i
            required_clearance = alpha_h_i / 2
            opti.subject_to(min_dist_to_obstacle > required_clearance)
            #opti.subject_to(norm_2(X[:, i] - o_pos) >= (o_rad + alpha_h[i]))
            #opti.subject_to(sumsqr(X[:, i] - o_pos) >= (o_rad + alpha_h[i])**2)
            #if i < N-1:  # Clearance constraints for midpoints
             #   midpoint = (X[:, i] + X[:, i+1]) / 2
                #opti.subject_to(norm_2(midpoint - o_pos) >= (o_rad + alpha_h[i]))
              #  opti.subject_to(sumsqr(midpoint - o_pos) >= (o_rad + alpha_h[i])**2)

    f = object_function_alpha(X, alpha_h, V, N)
    #f = object_function(X,N)
    
    opti.minimize(f)
 
    opts = {"verbose": True, "ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-1, "ipopt.constr_viol_tol": 1e-2, "expand": True, "ipopt.sb" : "yes"}  
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
        "verbose": False,  # sqpmethod verbosity
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

        sol_waypoints = sol.value(X)  # Extract the optimized waypoints if solution is feasible
        sol_alpha_h = sol.value(alpha_h)
        solver_stats = opti.stats()
        solver_time = solver_stats.get('t_proc_total', None)
        sol_V  =   sol.value(V)

        result_data = {

            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time,
            "sol_V": sol_V
        }

        result_queue.put(result_data)

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

        solver_stats = opti.stats()
        solver_time = solver_stats.get('t_proc_total', None)
        #Do something to get rid of error
        if P_sol is not None:
            sol_waypoints = P_sol[:N,:]
        else:
            sol_waypoints = P[:N,:]

        sol_alpha_h = np.full((N,), alpha_h0)  # Set alpha_h to a vector of alpha_h0 values
        sol_V = np.full((N,), v0)  # Set alpha_h to a vector of alpha_h0 values
        result_data = {
            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time,
            "sol_V": sol_V
        }
        result_queue.put(result_data)