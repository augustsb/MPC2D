from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs, norm_2
import numpy as np
import matplotlib.pyplot as plt
import traceback



#data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)

def object_function_alpha(X, alpha_h, V, N, cur_alpha_h):

    #coefficients_alpha = MX([0.0, -0.39706975, -7.71259023, 0.25644525, 22.35470889, 8.08533434])
    #intercept_alpha = MX(0.0859436060997259)

    intercept_alpha = MX([4.5667265])
    coefficients_alpha = MX([  0.0,   -14.98799005, -18.94456363,   8.3210305,   44.90256676, 13.24315094])

    #intercept_alpha =  MX([4.59732694])
    #coefficients_alpha = MX([0.0, -15.11378222, -18.90442604,  8.39899105,  44.96279529, 13.19096712])
    
    f_dist = MX(0)
    f_energy = MX(0)
    f_time = MX(0)
    f_initial_transition = MX(0)

    epsilon = 1e-6
    initial_transition_cost_factor = 5

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        #segment_length = norm_2(X[:, i+1] - X[:, i])
        f_dist += segment_length

        alpha_h_i = alpha_h[i]
        #V_i = V[i] + epsilon  # Add epsilon to avoid division by zero
        V_i = V[i]
        # Correct construction of polynomial features for iteration i
        linear_terms = vertcat(alpha_h_i,  V_i)
        squared_terms = vertcat(alpha_h_i**2,  V_i**2)
        interaction_terms = vertcat(alpha_h_i*V_i)
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept

        predicted_average_energy = intercept_alpha + dot(coefficients_alpha, all_terms)

 
        #predicted_time = segment_length / V_i**2
        #f_time += predicted_time

        f_energy +=  predicted_average_energy**2
       

    #f_initial_transition = initial_transition_cost_factor * ((alpha_h[0] - cur_alpha_h) ** 2)

    #total_cost = f_dist +  f_energy + f_time + f_initial_transition
    total_cost = f_dist + 100 * f_energy

    return total_cost

    #return f_energy

def object_function(X, N):

    f_dist = MX(0)

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        #segment_length = norm_2(X[:, i+1] - X[:, i])
        f_dist += segment_length

    return f_dist




def mpc_energy_efficiency_alpha(current_p, p_dot,  target, obstacles, params, controller_params, initial_N, k, result_queue, P):



    
    #alpha_h0 = controller_params['alpha_h']
    #cur_velocity = np.linalg.norm(p_dot)

    #min_velocity = 0.3
    #max_velocity = 1.0

    alpha_h_min = 5*np.pi/180
    alpha_h_max = 60*np.pi/180
    #alpha_h_min = 0.08
    #alpha_h_max = 0.50


    N = initial_N

    opti = Opti()  # Create an optimization problem


    X = opti.variable(3, N)  # Position variables
    #V = opti.variable(N)
    alpha_h = opti.variable(N)

    opti.subject_to(X[:, N-1] == P[N-1,:])
    opti.subject_to(X[:, 0] == P[0,:])

   
    for i in range(N):
        opti.subject_to(alpha_h[i] >= alpha_h_min)
        opti.subject_to(alpha_h[i] <= alpha_h_max)
        #opti.subject_to(V[i] >= min_velocity)
        #opti.subject_to(V[i] <= max_velocity)


    for i in range(N):  # Clearance constraints for waypoints
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            #opti.subject_to(norm_2(X[:, i] - o_pos) >= (o_rad + alpha_h[i]))
            opti.subject_to(sumsqr(X[:, i] - o_pos) >= (o_rad + alpha_h[i])**2)
            if i < N-1:  # Clearance constraints for midpoints
                midpoint = (X[:, i] + X[:, i+1]) / 2
                #opti.subject_to(norm_2(midpoint - o_pos) >= (o_rad + alpha_h[i]))
                opti.subject_to(sumsqr(midpoint - o_pos) >= (o_rad + alpha_h[i])**2)


    for i in range(N):
        opti.set_initial(X[:,i], P[i,:])

    #opti.set_initial(alpha_h[0], alpha_h_max)
        #opti.set_initial(V[0], cur_velocity)

    
    #f = object_function_alpha(X, alpha_h, V, N, alpha_h0)
    f = object_function(X,N)



    opti.minimize(f)
    #opti.minimize(f_dist)
 
    #Solver options
    opts = {"verbose": True,
             "ipopt.print_level": 1,
             "ipopt.max_iter": 10000,
             "ipopt.tol": 1e-2,
             "ipopt.constr_viol_tol": 1e-2, 
             "expand": True, 
             #"ipopt.linear_solver": "mumps",
             #"ipopt.linear_solver": "spral",
    }  
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
        #sol_V  =   sol.value(V)

        result_data = {

            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time,
            #"sol_V": sol_V
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