from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs, norm_2
import numpy as np
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from waypoint_methods import generate_initial_path, extend_horizon
import traceback


def object_function_all_params(X, alpha_h, omega_h, delta_h, V, N):

    coefficients = MX([0.00000000e+00, -3.76175305e-01, -1.39811782e-01, -3.33429081e-01,
                      -6.55247864e+00,  1.58587472e-01,  3.77745169e-02,  7.88736135e-02,
                       2.51218324e+01,  2.11785467e-02,  5.88215761e-02, 1.88560740e+00,
                       1.20102405e-01, -6.87588100e+00,  1.97807068e+00])
    
    intercept = MX(0.29915636699839293)
    
    f = 0

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        f += segment_length

        alpha_h_i = alpha_h[:, i]
        omega_h_i = omega_h[:, i]
        delta_h_i = delta_h[:, i]
        V_i = V[:, i]

        # Correct construction of polynomial features for iteration i
        linear_terms = vertcat(alpha_h_i, omega_h_i, delta_h_i, V_i)
        squared_terms = vertcat(alpha_h_i**2, omega_h_i**2, delta_h_i**2, V_i**2)
        interaction_terms = vertcat(alpha_h_i*omega_h_i, alpha_h_i*delta_h_i, alpha_h_i*V_i, omega_h_i*delta_h_i, omega_h_i*V_i, delta_h_i*V_i)
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept

        predicted_average_energy = intercept + dot(coefficients, all_terms) 
        #predicted_time = segment_length / fabs(V_i)
        f += predicted_average_energy 

    return f


def object_function_alpha(X, N):

    coefficients_alpha = MX([0.0, -0.39706975, -7.71259023, 0.25644525, 22.35470889, 8.08533434])
    intercept_alpha = MX(0.0859436060997259)
    
    f = 0

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point

        segment_length = sumsqr(X[:, i+1] - X[:, i])
        f += segment_length

        #poly_features = vertcat(1, alpha_h[i], V[i], alpha_h[i]**2, alpha_h[i]*V[i], V[i]**2)

        #predicted_average_energy = intercept_alpha + dot(coefficients_alpha, poly_features)
        #predicted_time = segment_length / fabs(V_i)
        #f += predicted_average_energy 
        
    return f




def mpc_energy_efficiency(current_p, p_dot,  target, obstacles, params, controller_params, initial_N, k, result_queue, P, all_params):


    min_velocity = 0.3
    max_velocity = 2.0
    alpha_h0 = controller_params['alpha_h']

    N = initial_N


    opti = Opti()  # Create an optimization problem

    X = opti.variable(3, N)  # Position variables
    alpha_h = opti.variable(1, N)


    if all_params:
        omega_h0 = controller_params['omega_h']
        delta_h0 = controller_params['delta_h']
        omega_h = opti.variable(1,N)
        delta_h = opti.variable(1,N)
        V = opti.variable(1, N) #Velocities
        opti.set_initial(delta_h[0], delta_h0)
        opti.set_initial(omega_h[0], omega_h0)
        opti.set_initial(V[0], 0.6)
        for i in range(N):
            opti.subject_to(40*np.pi/180 <= omega_h[i]) 
            opti.subject_to(omega_h[i] <= 210*np.pi/180) 
            opti.subject_to(30*np.pi/180 <= delta_h[i]) 
            opti.subject_to(delta_h[i] <= 90*np.pi/180) 
            opti.subject_to(min_velocity <= V[i]) 
            opti.subject_to(V[i] <= max_velocity)  # Maximum velocity constraint


    
    for i in range(N):
        opti.set_initial(X[:,i], P[i,:])
        opti.set_initial(alpha_h[i], alpha_h0)

    opti.subject_to(X[:, N-1] == P[N-1,:])
    opti.subject_to(X[:, 0] == P[0,:])

    for i in range(N):
        opti.subject_to(15*np.pi/180 <= alpha_h[i])
        opti.subject_to(alpha_h[i] <= 90*np.pi/180)


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

    # Minimize the objective
    if all_params:
        f_all_params = object_function_all_params(X, alpha_h, omega_h, delta_h, V, N)
        opti.minimize(f_all_params)

    else:
        f_alpha = object_function_alpha(X, N)
        opti.minimize(f_alpha)


    #Solver options
    opts = {"verbose": True, "ipopt.print_level": 1, "ipopt.max_iter": 1000, "ipopt.tol": 1e-2, "ipopt.constr_viol_tol": 1e-2, "expand": True,}  
    opti.solver('ipopt', opts)


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