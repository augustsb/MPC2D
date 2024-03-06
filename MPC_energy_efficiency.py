from casadi import Opti, sumsqr, sqrt, Function, dot, MX, mtimes, Function, fmax, vertcat, Callback, fabs
import numpy as np
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from waypoint_methods import redistribute_waypoints, generate_initial_path, extend_horizon
import traceback
from joblib import load
import torch



#Sklearn model
#model_rf = load('models/model_rf.joblib')

#Neural network
"""
# Load saved weights and biases
model_params = torch.load('model_params.pth')
# CasADi model construction
x = MX.sym('x', 2)
W1 = MX(model_params['weights_fc1'])
b1 = MX(model_params['bias_fc1'])
W2 = MX(model_params['weights_fc2'])
b2 = MX(model_params['bias_fc2'])
# Construct the neural network in CasADi
layer1 = mtimes(W1, x) + b1  # First linear layer
layer1_act = fmax(layer1, 0)  # ReLU activation
output = mtimes(W2, layer1_act) + b2  # Second linear layer
# Create a CasADi function for the neural network
nn_casadi = Function('nn', [x], [output])

"""

# Coeffecients for alpha, velocity for predicting power
coefficients_alpha = [0.0, -0.39706975, -7.71259023, 0.25644525, 22.35470889, 8.08533434]
intercept_alpha = 0.0859436060997259

#Coeffecients alpha, omega, delta, velocity for predicting power
#coeffecients =  [0.00000000e+00, -3.76175305e-01, -1.39811782e-01, -3.33429081e-01,
# -6.55247864e+00,  1.58587472e-01,  3.77745169e-02,  7.88736135e-02,
# 2.51218324e+01,  2.11785467e-02,  5.88215761e-02, 1.88560740e+00,
#  1.20102405e-01, -6.87588100e+00,  1.97807068e+00]

#intercepts = 0.29915636699839293






def mpc_energy_efficiency(current_p, p_dot,  target, obstacles, params, controller_params, initial_N, k, result_queue, P):

    coefficients = MX([0.00000000e+00, -3.76175305e-01, -1.39811782e-01, -3.33429081e-01,
                      -6.55247864e+00,  1.58587472e-01,  3.77745169e-02,  7.88736135e-02,
                       2.51218324e+01,  2.11785467e-02,  5.88215761e-02, 1.88560740e+00,
                       1.20102405e-01, -6.87588100e+00,  1.97807068e+00])
    
    intercept = MX(0.29915636699839293)
    

    min_velocity = 0.3
    max_velocity = 2.0
    #alpha_h0 = controller_params['alpha_h']
    #omega_h0 = controller_params['omega_h']
    #delta_h0 = controller_params['delta_h']

    N = min(initial_N, P.shape[0])


    opti = Opti()  # Create an optimization problem

    X = opti.variable(3, N)  # Position variables
    V = opti.variable(1, N) #Velocities
    alpha_h = opti.variable(1, N)
    omega_h = opti.variable(1,N)
    delta_h = opti.variable(1,N)
    #delta_h = controller_params['delta_h']



    energy_objective = 0
    total_distance_objective = 0

    for i in range(N-1):  # Iterate over the horizon, except the last point where there's no next point
        segment_length = sumsqr(X[:, i+1] - X[:, i])
        total_distance_objective += segment_length
        total_distance_objective +=   sumsqr(P[N-1,:] - X[:,i])

        #poly_features = vertcat(1, alpha_h[i], V[i], alpha_h[i]**2, alpha_h[i]*V[i], V[i]**2)
        #predicted_average_energy = intercept + dot(coefficients, poly_features)
        # Correct indexing for each variable at iteration i
        alpha_h_i = alpha_h[:, i]
        omega_h_i = omega_h[:, i]
        delta_h_i = delta_h[:, i]
        #delta_h_i = delta_h
        V_i = V[:, i]

        # Correct construction of polynomial features for iteration i
        linear_terms = vertcat(alpha_h_i, omega_h_i, delta_h_i, V_i)
        squared_terms = vertcat(alpha_h_i**2, omega_h_i**2, delta_h_i**2, V_i**2)
        interaction_terms = vertcat(alpha_h_i*omega_h_i, alpha_h_i*delta_h_i, alpha_h_i*V_i, omega_h_i*delta_h_i, omega_h_i*V_i, delta_h_i*V_i)
        all_terms = vertcat(1, linear_terms, squared_terms, interaction_terms)  # Include 1 for the intercept

        predicted_average_energy = dot(coefficients, all_terms) + intercept
        #predicted_time = segment_length / fabs(V_i)
        energy_objective += predicted_average_energy 



    # Initial position constraint
    #opti.subject_to(V[:, 0] == cur_velocity)
    #opti.set_initial(alpha_h[:,0], alpha_h0)
    #opti.set_initial(omega_h[:,0], omega_h0)
    #opti.set_initial(V[:,0], 0.5)
    #opti.set_initial(delta_h[:,0], delta_h0)

    opti.subject_to(X[:, N-1] == P[N-1,:])
    opti.subject_to(sumsqr(X[:, 0] - P[0,:]) <= 0.1)

    

    for i in range(N):
        opti.subject_to(V[:, i] >= min_velocity) 
        opti.subject_to(V[:, i] <= max_velocity)  # Maximum velocity constraint
        opti.subject_to(alpha_h[:, i] >= 10*np.pi/180) 
        opti.subject_to(alpha_h[:, i] <= 90*np.pi/180) 
        opti.subject_to(omega_h[:, i] >= 50*np.pi/180) 
        opti.subject_to(omega_h[:, i] <= 210*np.pi/180) 
        opti.subject_to(delta_h[:, i] >= 30*np.pi/180) 
        opti.subject_to(delta_h[:, i] <= 90*np.pi/180) 


    # Movement constraint between steps (simplified dynamics)
    for h in range(N-2):
        opti.subject_to(sumsqr(X[:, h+1] - X[:, h]) <= (k + 0.1) **2)
        #opti.subject_to(sumsqr(X[:, h+1] - X[:, h]) >= (k + -0.1) **2)

    for h in range(N-1):  # Clearance constraints for waypoints
        for obstacle in obstacles:
            o_pos = obstacle['center']
            o_rad = obstacle['radius']
            opti.subject_to(sumsqr(X[:, h] - o_pos) >= (o_rad + alpha_h[h])**2)
            if h < N-1:  # Clearance constraints for midpoints
                midpoint = (X[:, h] + X[:, h+1]) / 2
                opti.subject_to(sumsqr(midpoint - o_pos) >= (o_rad + alpha_h[h])**2)
 

    # Minimize the objective
    #opti.minimize(energy_objective + total_distance_objective)
    opti.minimize(10*energy_objective + total_distance_objective)

    # Solver options
    opts = {"verbose": True, "ipopt.print_level": 1, "ipopt.max_iter": 10000, "ipopt.tol": 1e-2, "ipopt.constr_viol_tol": 1e-2, "expand": True}
    opti.solver('ipopt', opts)


    try:
        sol = opti.solve()
        sol_waypoints = sol.value(X)  # Extract the optimized waypoints if solution is feasible
        sol_alpha_h = sol.value(alpha_h)
        sol_V = sol.value(V)
        sol_omega_h = sol.value(omega_h)
        sol_delta_h = sol.value(delta_h)
        #distributed_waypoints = redistribute_waypoints(sol_waypoints, k, current_p)  # Redistribute waypoints
        #result_queue.put((distributed_waypoints, sol_V, sol_alpha_h))
        result_queue.put((sol_waypoints, sol_alpha_h, sol_omega_h, sol_delta_h, sol_V))

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