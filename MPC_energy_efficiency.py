from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs, norm_2
import numpy as np
import matplotlib.pyplot as plt
from obstacle_methods import calculate_min_dist_to_obstacle
import traceback



#data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)


def object_function_all_params(X, alpha_h, omega_h, delta_h, V, N):
  
    """
    intercept =  MX([-9.61370929])
    coefficients = MX([0.0, 6.37632023,  -2.13277937,  34.07197804,   6.43777459,
   12.1279542,   14.34658138, -51.53449861,   0.50842533,   0.95492672,
   -7.38813161,   2.96602627,  -4.80776931, -24.59936993,  -6.44171684])
   """


    intercept = MX([-2.00870276])
    coefficients = MX([0.00000000e+00,  4.17084787e+00,  1.30150304e+00,  2.83590215e-01,
                    -1.70834333e+00,  5.06018915e+00,  3.45767527e+00, -1.59892163e+01,
                     1.90201899e+01,  5.79456468e-03, -2.54774299e+00,  3.02757522e+00,
                     6.22982835e+00, -1.52761404e+01, -4.92614971e+00])
    

    intercept = MX([-7.83116369])
    coefficients =  MX([4.73044527e-10,  6.09891136e+00, 7.88693391e-01,  3.38424818e+01,
  -9.90377855e+00, 9.96133136e+00,  8.38996787e+00, -4.03914370e+01,
   7.63690483e+00, -4.90251301e-03, -5.50151823e+00,  3.68921190e+00,
  -3.54562534e+01, 1.38939097e+01,-3.16562456e+00, -2.03963050e+00,
   4.37586051e+00, -1.19442052e+01,  1.19440567e+01,  1.15498551e+00,
  -1.23751616e+01,  7.75240567e+00,  4.13014253e+01, -6.52288791e+01,
  -5.75669291e-01,  7.71848867e-02, -5.25832652e-01, -9.53014958e-01,
   6.03702143e+00, -4.59527939e+00,  2.69215876e+00,  7.32832244e+00,
   1.14169391e+01,  1.97269179e+00,-3.84810697e+00])


    
    f_dist = MX(0)
    f_energy = MX(0)
    f_time = MX(0)
    epsilon = 1e-6

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        f_dist += segment_length

        alpha_h_i = alpha_h[i]
        omega_h_i = omega_h[i]
        delta_h_i = delta_h[i]
        #V_i = V[i] + epsilon  # Add epsilon to avoid division by zero
        V_i = V[i] 
        
        # Correct construction of polynomial features for iteration i
        #linear_terms = vertcat(alpha_h_i, omega_h_i, delta_h_i, V_i)
        #squared_terms = vertcat(alpha_h_i**2, omega_h_i**2, delta_h_i**2, V_i**2)
        #interaction_terms = vertcat(alpha_h_i*omega_h_i, alpha_h_i*delta_h_i, alpha_h_i*V_i, omega_h_i*delta_h_i, omega_h_i*V_i, delta_h_i*V_i)

        # Constructing all terms
    # Constructing terms
        linear_terms = vertcat(alpha_h_i, omega_h_i, delta_h_i, V_i)
        squared_terms = vertcat(alpha_h_i**2, omega_h_i**2, delta_h_i**2, V_i**2)
        interaction_terms_2nd = vertcat(alpha_h_i*omega_h_i, alpha_h_i*delta_h_i, alpha_h_i*V_i, omega_h_i*delta_h_i, omega_h_i*V_i, delta_h_i*V_i)
        cubic_terms = vertcat(alpha_h_i**3, omega_h_i**3, delta_h_i**3, V_i**3)
        squared_interactions = vertcat(alpha_h_i**2*omega_h_i, alpha_h_i*omega_h_i**2, alpha_h_i**2*delta_h_i, alpha_h_i*delta_h_i**2, alpha_h_i**2*V_i, alpha_h_i*V_i**2,
                                    omega_h_i**2*delta_h_i, omega_h_i*delta_h_i**2, omega_h_i**2*V_i, omega_h_i*V_i**2, delta_h_i**2*V_i, delta_h_i*V_i**2)
        three_feature_interactions = vertcat(alpha_h_i*omega_h_i*delta_h_i, alpha_h_i*omega_h_i*V_i, alpha_h_i*delta_h_i*V_i, omega_h_i*delta_h_i*V_i)

        # Combine all terms, including 1 for the intercept
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms_2nd, cubic_terms, squared_interactions, three_feature_interactions)
        
        #all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept
        predicted_average_energy = intercept + dot(coefficients, all_terms) 
        
        #predicted_time = segment_length / V_i
        f_time += 0

        f_energy += predicted_average_energy

    return f_dist, f_energy, f_time
    #return f_energy



def mpc_energy_efficiency(current_p, p_dot,  target, obstacles, params, controller_params, initial_N, k, result_queue, P, P_sol):

    alpha_h0 = controller_params['alpha_h']
    omega_h0 = controller_params['omega_h']
    delta_h0 = controller_params['delta_h']
    #safe_margin = 0.1


    min_velocity = controller_params['v_min']
    max_velocity = controller_params['v_max']
    alpha_h_min = controller_params['alpha_h_min']
    alpha_h_max = controller_params['alpha_h_max']
    omega_h_min = 80*np.pi/180
    omega_h_max = 210*np.pi/180
    delta_h_min = 20*np.pi/180
    delta_h_max = 60*np.pi/180

    N = initial_N

    opti = Opti()  # Create an optimization problem


    X = opti.variable(3, N)  # Position variables
    V = opti.variable(N)
    alpha_h = opti.variable(N)
    omega_h = opti.variable(N)
    delta_h = opti.variable(N)


    opti.subject_to(X[:, 0] == current_p)
    opti.subject_to(X[:, N-1] == target)


    for i in range(N):
        opti.subject_to(opti.bounded(alpha_h_min, alpha_h[i], alpha_h_max))
        opti.subject_to(opti.bounded(omega_h_min, omega_h[i], omega_h_max))
        opti.subject_to(opti.bounded(delta_h_min, delta_h[i], delta_h_max))
        opti.subject_to(opti.bounded(min_velocity, V[i], max_velocity))


    for i in range(N-1):  # Clearance constraints for waypoints
        A = X[:, i]
        B = X[:, i+1]
        alpha_h_i = alpha_h[i]
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            min_dist_to_obstacle = calculate_min_dist_to_obstacle(A, B, o_pos, o_rad)
            opti.subject_to(min_dist_to_obstacle > alpha_h_i/2)
            #opti.subject_to(norm_2(X[:, i] - o_pos) >= (o_rad + alpha_h[i]))
            #opti.subject_to(sumsqr(X[:, i] - o_pos) > (o_rad + alpha_h[i])**2)
            #if i < N-1:  # Clearance constraints for midpoints
                #midpoint = (X[:, i] + X[:, i+1]) / 2
                #opti.subject_to(norm_2(midpoint - o_pos) >= (o_rad + alpha_h[i]))
                #opti.subject_to(sumsqr(midpoint - o_pos) > (o_rad + alpha_h[i])**2)
            


    opti.set_initial(X[:,0], P[:,0])
    opti.set_initial(X[:,N-1], target)

    if (P_sol is not None):
        for i in range(1, N-1):
            opti.set_initial(X[:,i], P_sol[:,i])

    else:
        for i in range(N):
            opti.set_initial(X[:,i], P[:,i])


    f_dist, f_energy, f_time = object_function_all_params(X, alpha_h, omega_h, delta_h, V, N)

    opti.minimize(f_dist + f_energy + 1000*f_time)


    opts = {"verbose": True, "ipopt.print_level": 0, "ipopt.max_iter": 1000, "ipopt.tol": 1e-3, "ipopt.constr_viol_tol": 1e-3, "expand": True, "ipopt.sb" : "yes"}  
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

        result_data = {

            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time
        }

        sol_omega_h = sol.value(omega_h)
        sol_delta_h = sol.value(delta_h)
        #sol_delta_h = np.full((N,), delta_h)  # Set alpha_h to a vector of alpha_h0 values
        sol_V  =   sol.value(V)
        # Adding additional parameters only if all_params is True
        result_data.update({
                "sol_omega_h": sol_omega_h,
                "sol_delta_h": sol_delta_h,
                "sol_V": sol_V,
        })


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
        sol_waypoints = P[:N,:]
        sol_alpha_h = np.full((N,), alpha_h0)  # Set alpha_h to a vector of alpha_h0 values
        sol_omega_h = np.full((N,), omega_h0)  # Set alpha_h to a vector of alpha_h0 values
        sol_delta_h = np.full((N,), delta_h0)  # Set alpha_h to a vector of alpha_h0 values
        sol_V = np.full((N,), 0)  # Set alpha_h to a vector of alpha_h0 values
        result_data = {
            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time,
             "sol_V": sol_V
        }
        result_data.update({
                "sol_omega_h": sol_omega_h,
                "sol_delta_h": sol_delta_h,
        })
        result_queue.put(result_data)