from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs, norm_2
import numpy as np
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from waypoint_methods import generate_initial_path, extend_horizon
from predict_energy import load_and_preprocess_data, find_optimal_configuration
import traceback


coefficients_alpha = MX([0.0, -0.39706975, -7.71259023, 0.25644525, 22.35470889, 8.08533434])
intercept_alpha = MX(0.0859436060997259)


data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)



def object_function_all_params_lookup(X,  N,  alpha_h,  V_min):

    f = 0

    for i in range(N-1):
        segment_length = sumsqr(X[:, i+1] - X[:, i])

        alpha_h_i = alpha_h[i]
        
        # Retrieve the optimal entry from your dataset
        optimal_entry = find_optimal_configuration(data_all, alpha_h_i, V_min)

        if optimal_entry is not None:
            # Use the energy value from the optimal configuration
            predicted_average_energy = optimal_entry['average_energy']
            predicted_average_velocity = optimal_entry['average_velocity']

            # Calculate predicted time for segment
            predicted_time = segment_length / predicted_average_velocity
            # Update the objective
            f += predicted_average_energy * predicted_time
        else:

            break

    return f


def object_function_all_params(X, alpha_h, omega_h, delta_h, V, N):

    coefficients = MX([0.00000000e+00, -3.76175305e-01, -1.39811782e-01, -3.33429081e-01,
                      -6.55247864e+00,  1.58587472e-01,  3.77745169e-02,  7.88736135e-02,
                       2.51218324e+01,  2.11785467e-02,  5.88215761e-02, 1.88560740e+00,
                       1.20102405e-01, -6.87588100e+00,  1.97807068e+00])
    
    intercept = MX(0.29915636699839293)
    
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
        linear_terms = vertcat(alpha_h_i, omega_h_i, delta_h_i, V_i)
        squared_terms = vertcat(alpha_h_i**2, omega_h_i**2, delta_h_i**2, V_i**2)
        interaction_terms = vertcat(alpha_h_i*omega_h_i, alpha_h_i*delta_h_i, alpha_h_i*V_i, omega_h_i*delta_h_i, omega_h_i*V_i, delta_h_i*V_i)
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept

        predicted_average_energy = intercept + dot(coefficients, all_terms) 
 
        #predicted_time = segment_length / V_i
        #f_time += predicted_time

        f_energy += predicted_average_energy 

    return f_dist, f_energy
    #return f_energy



def mpc_energy_efficiency(current_p, p_dot,  target, obstacles, params, controller_params, initial_N, k, result_queue, P, all_params):


    
    alpha_h0 = controller_params['alpha_h']
    omega_h0 = controller_params['omega_h']
    delta_h0 = controller_params['delta_h']
    cur_velocity = np.linalg.norm(p_dot)

    #Pareto front
    """ 
    min_velocity = 0.2
    max_velocity = 0.96
    alpha_h_min = 0.08
    alpha_h_max = 0.26
    omega_h_min = 2.66
    omega_h_max = 3.66
    delta_h_min = 0.08
    delta_h_max = 0.70
    """
   

    min_velocity = 0.3
    max_velocity = 0.7
    alpha_h_min = 5*np.pi/180
    alpha_h_max = 90*np.pi/180
    omega_h_min = 40*np.pi/180
    omega_h_max = 210*np.pi/180
    delta_h_min = 30*np.pi/180
    delta_h_max = 90*np.pi/180

    N = initial_N

    opti = Opti()  # Create an optimization problem


    X = opti.variable(3, N)  # Position variables
    V = opti.variable(N)
    alpha_h = opti.variable(N)
    omega_h = opti.variable(N)
    delta_h = opti.variable(N)


    opti.subject_to(X[:, N-1] == P[N-1,:])
    opti.subject_to(X[:, 0] == P[0,:])
    for i in range(N):
        opti.subject_to(alpha_h[i] >= alpha_h_min)
        opti.subject_to(alpha_h[i] <= alpha_h_max)
        opti.subject_to(omega_h[i] >= omega_h_min)
        opti.subject_to(omega_h[i] <= omega_h_max)
        opti.subject_to(delta_h[i] >= delta_h_min)
        opti.subject_to(delta_h[i] <= delta_h_max)
        opti.subject_to(V[i] >= min_velocity)
        opti.subject_to(V[i] <= max_velocity)
        #if i < N-1:
            #opti.subject_to(sumsqr(X[:, i] - X[:, i+1]) > 0)


    for i in range(N):  # Clearance constraints for waypoints
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            #opti.subject_to(norm_2(X[:, i] - o_pos) >= (o_rad + alpha_h[i]))
            opti.subject_to(sumsqr(X[:, i] - o_pos) > (o_rad + alpha_h[i])**2)
            if i < N-1:  # Clearance constraints for midpoints
                midpoint = (X[:, i] + X[:, i+1]) / 2
                #opti.subject_to(norm_2(midpoint - o_pos) >= (o_rad + alpha_h[i]))
                opti.subject_to(sumsqr(midpoint - o_pos) > (o_rad + alpha_h[i])**2)


    for i in range(N):
        opti.set_initial(X[:,i], P[i,:])

    opti.set_initial(alpha_h[0], alpha_h0)
    opti.set_initial(omega_h[0], omega_h0)
    opti.set_initial(delta_h[0], delta_h0)




    f_dist, f_energy = object_function_all_params(X, alpha_h, omega_h, delta_h, V, N)
    
    #f_energy = object_function_all_params(X, alpha_h, omega_h, delta_h, V, N)
    #f = object_function_all_params_lookup(X,  N,  alpha_h,  min_velocity)
    #f = object_function(X, N)

    opti.minimize(f_dist + f_energy)
    #opti.minimize(10 * f_energy)
 

    #Solver options
    opts = {"verbose": True,
             "ipopt.print_level": 1,
             "ipopt.max_iter": 1000,
             "ipopt.tol": 1e-2,
             "ipopt.constr_viol_tol": 1e-3, 
             "expand": True, 
             "ipopt.linear_solver": "mumps",
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

        result_data = {

            "sol_waypoints": sol_waypoints,
            "sol_alpha_h": sol_alpha_h,
            "solver_time": solver_time
        }



        if all_params:
            sol_omega_h = sol.value(omega_h)
            sol_delta_h = sol.value(delta_h)
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