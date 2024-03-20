from casadi import MX,  vertcat, integrator, Function
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from init_controller_parameters import init_controller_parameters
from calculate_v_dot_MPC import calculate_v_dot_MPC
from draw_snake_robot import draw_snake_robot
from extract_states import extract_states
from init_waypoint_parameters import init_waypoint_parameters
from calculate_pathframe_state import calculate_pathframe_state
import numpy as np
from obstacle_methods import  filter_obstacles, analyze_future_path
from MPC_shortest_path import mpc_shortest_path
from MPC_energy_efficiency import mpc_energy_efficiency
from MPC_energy_efficiency_alpha import mpc_energy_efficiency_alpha
import threading
from queue import Queue, Empty  # Note the import of Empty here
from path_methods import  generate_initial_path, extend_horizon, expand_initial_guess
import json
import os
import traceback
from visualize_simulation_results import visualize_simulation_results, visualize_simulation_results_3d, convert_to_serializable, plot_colored_path
from predict_energy import load_and_preprocess_data, load_and_preprocess_data_single, find_optimal_configuration, load_and_preprocess_data_json, data_pareto_2802, data_pareto_1503
from change_gait_params import calculate_coeffs_list, calculate_start_conditions, calculate_end_conditions
import pandas as pd



#data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)
#Reprocessed
data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/reprocessed_results_2802", "reprocessed_chunk_results_", 16)
data_all_predicted =  load_and_preprocess_data("/home/augustsb/MPC2D/predictions_reprocessed_results_2802", "predicted_reprocessed_chunk_results__", 16)
data_1703_path = "/home/augustsb/MPC2D/results_1703/simulation_results.json"
data_1703 = load_and_preprocess_data_json(data_1703_path)
#data_combined = pd.concat([data, data_new, data_1503], ignore_index=True)
data_new = pd.concat([data_1703])  # Use ignore_index=Tr
data_new = data_new[data_new['average_energy'] <= 20]
data_delta_20 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_20.csv", 1)
data_delta_40 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_40.csv", 1)
data_delta_30 = load_and_preprocess_data_single("/home/augustsb/MPC2D/results_1803", "simulation_results_delta_30.csv", 1)
data_delta_sorted = load_and_preprocess_data_single("/home/augustsb/MPC2D/", "data_sorted_by_average_energy.csv", 1)

columns = ['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'average_energy']
data_lateral_1503 = pd.DataFrame(data_pareto_1503, columns=columns)
data_lateral_2802 = pd.DataFrame(data_pareto_2802, columns=columns)

data_lateral_1503.columns = columns
data_lateral_2802.columns = columns

# Now, when you concatenate, the resulting DataFrame will have the columns named:
data_combined = pd.concat([data_lateral_1503, data_lateral_2802], ignore_index=True)



def start_simulation(mode, dimension):
    # Initialize parameters

    params = init_model_parameters()

    n = params['n']
    l = params['l']

    controller_params = init_controller_parameters(n,l)

    y_int = controller_params['y_int0']
    z_int = controller_params['z_int0']
    theta_z = params['theta_z0']
    theta_x0_dot = params['theta_x0_dot']
    theta_z0_dot = params['theta_z0_dot']
    p_CM = params['p_CM0']
    p_CM_dot = params['p_CM0_dot'] 

    

    #num_obstacles = 8
    #rea_size = (31, 8)
    #min_distance_to_start_target = 2.0
    #obstacles = generate_random_obstacles(num_obstacles, area_size, p_CM, target, min_distance_to_start_target)
    """
    obstacles = [
        {'center': (15, 1.8, 0), 'radius': 1.60},  # First obstacle
        {'center': (15, -1.8, 0), 'radius': 1.60}, # Second obstacle
    ]
    """


    if dimension == '2D':
        #1 obstacle
        #target = np.array([20.0 , 0.0, 0.0]) #2D
        #obstacles = [{'center': (10, 0, 0), 'radius': 1.5},]

        #8 obstacles
        """
        target = np.array([29.0 , 0.0, 0.0]) #2D   
        obstacles = [
            {'center': (18.0, -1.0, 0), 'radius': 2.0},  # o0
            {'center': (10.0, 0.0, 0), 'radius': 1.5},   # o1
            {'center': (6.0, 1.0, 0), 'radius': 1.0},    # o2
            {'center': (5.0, 4.0, 0), 'radius': 2.0},    # o3
            {'center': (5.0, -4.0, 0), 'radius': 2.0},   # o4
            {'center': (25.0, 4.0, 0), 'radius': 2.0},   # o5
            {'center': (25.0, -4.0, 0), 'radius': 2.0},  # o6
            {'center': (15.0, 4.0, 0), 'radius': 1.0},   # o7
            {'center': (15.0, -4.0, 0), 'radius': 1.0}   # o8
        ]
        """

        #14 obstacles
        target = np.array([29.0 , 0.0, 0.0]) #2D
        obstacles = [
                    {'center': (18.0, -1.0, 0), 'radius': 2.0},  # o0
                    {'center': (10.0, 0.0, 0), 'radius': 1.5},   # o1
                    {'center': (6.0, 1.0, 0), 'radius': 1.0},    # o2
                    {'center': (5.0, 4.0, 0), 'radius': 2.0},    # o3
                    {'center': (5.0, -4.0, 0), 'radius': 2.0},   # o4
                    {'center': (25.0, 4.0, 0), 'radius': 2.0},   # o5
                    {'center': (25.0, -4.0, 0), 'radius': 2.0},  # o6
                    {'center': (15.0, 4.0, 0), 'radius': 1.0},   # o7
                    {'center': (15.0, -4.0, 0), 'radius': 1.0},  # o8
                    {'center': (10.0, -5.0, 0), 'radius': 1.5},  # o9
                    {'center': (20.0, -6.0, 0), 'radius': 1.0},  # o10
                    {'center': (22.0, 0, 0),    'radius': 0.5},  # o11
                    {'center': (15.0, -8.0, 0), 'radius': 2.0},  # o12
                    {'center': (20.0, 4.0, 0), 'radius': 1.5},  # o13
                    {'center': (25.0, -0.5, 0), 'radius': 1.0},  # o14
        ]
      


    if dimension == '3D':

        #1 obstacle
        
        """
        target = np.array([20.0 , 0.0, 0.0]) #2D
        obstacles = [{'center': (10, 0, 0), 'radius': 1.5},]
        """
        

        #8 obstacles


        target = np.array([29.0 , 0.0, 6.0]) #3D
        obstacles = [
            {'center': (18.0, -1.0, 4.5), 'radius': 2.0},  # o0
            {'center': (10.0, 0.0, 1.8), 'radius': 1.5},   # o1
            {'center': (6.0, 1.0, 3.0), 'radius': 1.0},    # o2
            {'center': (5.0, 4.0, 4.0), 'radius': 2.0},    # o3
            {'center': (5.0, -4.0, 4.0), 'radius': 2.0},   # o4
            {'center': (25.0, 4.0, 4.0), 'radius': 2.0},   # o5
            {'center': (25.0, -4.0, 4.0), 'radius': 2.0},  # o6
            {'center': (15.0, 4.0, 2.0), 'radius': 1.0},   # o7
            {'center': (15.0, -4.0, 2.0), 'radius': 1.0}   # o8
        ]

        """

        #14 obstacles
        target = np.array([29.0 , 0.0, 6.0])
        obstacles = [
                    {'center': (18.0, -1.0, 4.5), 'radius': 2.0},  # o0
                    {'center': (10.0, 0.0, 1.8), 'radius': 1.5},   # o1
                    {'center': (6.0, 1.0, 3.0), 'radius': 1.0},    # o2
                    {'center': (5.0, 4.0, 4.0), 'radius': 2.0},    # o3
                    {'center': (5.0, -4.0, 4.0), 'radius': 2.0},   # o4
                    {'center': (25.0, 4.0, 4.0), 'radius': 2.0},   # o5
                    {'center': (25.0, -4.0, 4.0), 'radius': 2.0},  # o6
                    {'center': (15.0, 4.0,4.0), 'radius': 1.0},   # o7
                    {'center': (15.0, -4.0, 2.0), 'radius': 1.0},  # o8
                    {'center': (10.0, -5.0, 2.0), 'radius': 1.5},  # o9
                    {'center': (20.0, -6.0, 3.0), 'radius': 1.0},  # o10
                    {'center': (22.0, 0, 3.0),    'radius': 0.5},  # o11
                    {'center': (15.0, -8.0, 2.0), 'radius': 2.0},  # o12
                    {'center': (20.0, 4.0, 8.0), 'radius': 1.5},  # o13
                    {'center': (25.0, -0.5, 6.0), 'radius': 1.0},  # o14
                    {'center': (22.0, 1, 6.0), 'radius': 1.0},  # o15
                    {'center': (15.0, 0.5, 4.0), 'radius': 1.0},  # o16
                    {'center': (15.0, -1, 2.0), 'radius': 1.0},  # o17
                    {'center': (25, 2.5, 6.0), 'radius': 1.0},  # o18
        ]
        """



     

    dt = 0.05  #Update frequency simulation
    mpc_dt = 0.5 #Update frequency mpc
    mpc_gait_dt = 2.0
    k = 1 #desired step length
    initial_N = 10 # Initial prediction horizon
    N = initial_N
    N_min = 2
    V_min = 0.4 #initial
    V_max = 1.2 #initial
    alpha_h_max = 90*np.pi/180
    alpha_h_min = 5*np.pi/180
    controller_params['v_min'] = V_min
    controller_params['v_max'] = V_max
    controller_params['alpha_h_min'] = alpha_h_min
    controller_params['alpha_h_max'] = alpha_h_max



    simulation_over = False
    t = 0


    total_distance_traveled = 0
    p_CM_previous = p_CM
    p_CM_list = []
    #middle_link_list = []
    all_link_x = []
    all_link_y = []
    all_link_z = []
    theta_x_list = []
    theta_z_list = []
    phi_ref_x_list = []
    energy_list = []
    velocity_list = []
    tot_energy = 0
    tot_solver_time = 0
    num_mpc_solutions = 0
    num_measurements = 0


    next_mpc_update_time = mpc_dt
    next_mpc_gait_update_time = mpc_dt
    mpc_start_time = 0
    T = 1.0 #transition time between gaits
    controller_params['T'] = T #transition time between gaits
    mpc_active = True
    controller_params['transition_in_progress'] = False

 
  
    P = generate_initial_path(p_CM, target, k)
    P = np.squeeze(P)
    if (N > P.shape[0] - 1): 
         N = P.shape[0] - 1
    initial_N = N
    N = extend_horizon(P, N, obstacles, P.shape[0], controller_params)
    goal = P[N-1,:]


    """
    start_point = tuple(p_CM.tolist())  # Convert p_CM (if it's a numpy array) to tuple
    goal_point = tuple(P[N-1, :].tolist())  # Convert P[N-1, :] to tuple
    P = rrt(start_point, goal_point, obstacles, controller_params['alpha_h'])
    P = interpolate_path(P, N)
    """

    if mpc_active:

        result_queue = Queue()

        if (mode == 'Distance'):
            mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM, target, obstacles, params, controller_params,
                                                                        N, k, result_queue, P.T, None))
            mpc_thread.start()
            try:
                P_sol, solver_time = result_queue.get()  # This will block until a solution is available
                tot_solver_time += solver_time
                num_mpc_solutions += 1
            except:
                print("Failed to generate initial waypoints")

        
        if (mode == 'Energy_alpha'):

            mpc_thread = threading.Thread(target=mpc_energy_efficiency_alpha, args=(p_CM, p_CM_dot, target, obstacles, params,
                                                                            controller_params, N, k, result_queue, P.T, None))
            mpc_thread.start()
            
            try:
                result = result_queue.get()  # This will block until a solution is available

                # Retrieve common results
                P_sol = result["sol_waypoints"]
                sol_alpha_h = result["sol_alpha_h"]
                solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set
                #sol_V = result["sol_V"]
                # Update controller params based on the retrieved solution
                controller_params.update({'alpha_h': sol_alpha_h[0]})  # Example for alpha_h

            except:
                print("Failed to generate initial waypoints")


        
        if (mode == 'Energy'):
            mpc_thread = threading.Thread(target=mpc_energy_efficiency, args=(p_CM, p_CM_dot, target, obstacles, params,
                                                                            controller_params, N, k, result_queue, P.T, None))
            mpc_thread.start()
            
            try:
                result = result_queue.get()  # This will block until a solution is available

                # Retrieve common results
                P_sol = result["sol_waypoints"]
                sol_alpha_h = result["sol_alpha_h"]
                solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set
                sol_V = result["sol_V"]
                # Update controller params based on the retrieved solution
                controller_params.update({'alpha_h': sol_alpha_h[0]})  # Example for alpha_h
                sol_omega_h = result["sol_omega_h"]
                sol_delta_h = result["sol_delta_h"]
                controller_params.update({'omega_h': sol_omega_h[0], 'delta_h': sol_delta_h[0]})

                # Aggregate solver time and count
                tot_solver_time += solver_time
                num_mpc_solutions += 1
            except:
                print("Failed to generate initial waypoints")


        waypoint_params = init_waypoint_parameters(P_sol.T)
        prev_solution = P_sol.T

    
    else:

        waypoint_params = init_waypoint_parameters(P[:N,:])


    #waypoint_params = init_waypoint_parameters(P_sol.T)
    waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)
    cur_alpha_path = waypoint_params['cur_alpha_path']
    cur_gamma_path = waypoint_params['cur_gamma_path']
    theta_x0 = np.full(n, cur_alpha_path)

    if dimension == '3D':
        theta_z0 = np.full(n,cur_gamma_path)
        v0 = vertcat(theta_x0, theta_z0, p_CM, theta_x0_dot, theta_z0_dot, p_CM_dot,  y_int, z_int)
        plt.ion()  # Turn on interactive plotting mode
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    if dimension == '2D':
        v0 = vertcat(theta_x0, p_CM, theta_x0_dot, p_CM_dot, y_int)
        plt.ion()  # Turn on interactive plotting mode
        fig, ax = plt.subplots()

    result_queue = Queue()
    mpc_thread = None
    
    try: 
        while not simulation_over:
            
            current_time = t

            if (current_time >= mpc_start_time):
                mpc_active = True

            # Check if it's time to start or update the MPC calculation
            if (current_time >= next_mpc_update_time and mpc_active and N > N_min):
             
                if mpc_thread is None or not mpc_thread.is_alive():
                
                    P = generate_initial_path(p_CM, target, k)
                    P = np.squeeze(P)
                    N = extend_horizon(P, N, obstacles, P.shape[0], controller_params)
                    goal = P[N-1,:]
                    prev_solution = expand_initial_guess(prev_solution, N, goal)
                    filtered_obstacles = filter_obstacles(p_CM, obstacles, N, goal)

                    """
                    path_condition = analyze_future_path(prev_solution, filtered_obstacles, 3)
                    if path_condition == "clear":
                        controller_params.update({'v_max': 1.2})  # Increase v_min for clear paths
                        #controller_params.update({'v_max': 3.0})  # Increase v_min for clear paths
                        controller_params.update({'alpha_h_max': 90*np.pi/180})  # Increase v_min for clear paths
                    else:
                        controller_params.update({'v_max': 0.5})  # Increase v_min for clear paths
                        #controller_params.update({'v_max': 0.6})  # Increase v_min for clear paths
                        controller_params.update({'alpha_h_max': 30*np.pi/180})  # Increase v_min for clear paths
                    """


                    if (mode == 'Energy'):
                        mpc_thread = threading.Thread(target=mpc_energy_efficiency(p_CM, p_CM_dot,  goal, filtered_obstacles, params, controller_params,
                                                                                     N, k, result_queue, P.T, prev_solution.T))
                        
                    if (mode == 'Energy_alpha'):
                        mpc_thread = threading.Thread(target=mpc_energy_efficiency_alpha(p_CM, p_CM_dot,  goal, filtered_obstacles, params, controller_params,
                                                                                     N, k, result_queue, P.T, prev_solution.T))

                    elif (mode == 'Distance'):
                        mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM, goal, filtered_obstacles, params, controller_params,
                                                                                       N, k,  result_queue, P.T, prev_solution.T))

                    mpc_thread.start()
                    next_mpc_update_time += mpc_dt  # Schedule next update
                    
            # Try to get the MPC results without blocking
            try:
                
                if (mode == 'Energy'):
   
                    result = result_queue.get_nowait()  # This will block until a solution is available
 
                    P_sol = result["sol_waypoints"]
                    sol_alpha_h = result["sol_alpha_h"]
                    solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set
                    sol_V = result["sol_V"]
                    sol_omega_h = result["sol_omega_h"]
                    sol_delta_h = result["sol_delta_h"]
                    # Update controller params based on the retrieved solution
                    #controller_params.update({'alpha_h': sol_alpha_h[0], 'omega_h': sol_omega_h[0], 'delta_h': sol_delta_h[0]})
                    print("sol_delta_h:", sol_delta_h[0])
                    print("sol_omega_h:", sol_omega_h[0])
                    print("sol_alpha_h:", sol_alpha_h[0])
                    print("sol_V:", sol_V[0])
                    if (current_time >= next_mpc_gait_update_time):

                        optimal_params = {'omega_h': sol_omega_h[0], 'delta_h': sol_delta_h[0], 'alpha_h': sol_alpha_h[0]}
                        # Example variables - these should be defined based on your specific needs
                        #alpha_h_current, omega_h_current, delta_h_current = controller_params['alpha_h'], controller_params['omega_h'], controller_params['delta_h']
                        alpha_h_target, omega_h_target, delta_h_target = optimal_params['alpha_h'], optimal_params['omega_h'], optimal_params['delta_h']
                        controller_params.update({'target_alpha_h': alpha_h_target, 'target_omega_h': omega_h_target, 'target_delta_h': delta_h_target})

                        #start_conditions = calculate_start_conditions(alpha_h_current, omega_h_current, delta_h_current, n, t)
                        #end_conditions = calculate_end_conditions(alpha_h_target, omega_h_target, delta_h_target, n, t, T)
   
                        #coeffs_list = calculate_coeffs_list(start_conditions, end_conditions, T)
                        #controller_params['coeffs_list'] = coeffs_list
                        controller_params.update({'transition_start_time': t})
                        controller_params.update({'transition_in_progress': True})

                        next_mpc_gait_update_time += mpc_gait_dt
                        #controller_params.update({'alpha_h': alpha_h_target, 'omega_h': omega_h_target, 'delta_h': delta_h_target})

                
                if (mode == 'Energy_alpha'):

                    result = result_queue.get_nowait()  # This will block until a solution is available
 
                    P_sol = result["sol_waypoints"]
                    sol_alpha_h = result["sol_alpha_h"]
                    sol_V = result["sol_V"]
                    solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set
                    print('Sol alpha:', sol_alpha_h[0])
                    print('sol_V:', sol_V[0])
                    if (current_time >= next_mpc_gait_update_time):

                        optimal_entry = find_optimal_configuration(data_delta_40, sol_alpha_h[0], controller_params['v_min'], controller_params['v_max'], sol_V[0])

                        optimal_params = {'omega_h': optimal_entry['omega_h'], 'delta_h': optimal_entry['delta_h'], 'alpha_h': optimal_entry['alpha_h']}
                        # Example variables - these should be defined based on your specific needs
                        alpha_h_current, omega_h_current, delta_h_current = controller_params['alpha_h'], controller_params['omega_h'], controller_params['delta_h']
                        alpha_h_target, omega_h_target, delta_h_target = optimal_params['alpha_h'], optimal_params['omega_h'], optimal_params['delta_h']
                        controller_params.update({'target_alpha_h': alpha_h_target, 'target_omega_h': omega_h_target, 'target_delta_h': delta_h_target})
                        
    
                        #start_conditions = calculate_start_conditions(alpha_h_current, omega_h_current, delta_h_current, n, t)
                        #end_conditions = calculate_end_conditions(alpha_h_target, omega_h_target, delta_h_target, n, t, T)
                        #coeffs_list = calculate_coeffs_list(start_conditions, end_conditions, T)
                        #controller_params.update({'coeffs_list':  coeffs_list})
                        controller_params.update({'transition_start_time': t})
                        controller_params.update({'transition_in_progress': True})
                    
                        next_mpc_gait_update_time += mpc_gait_dt
                        #controller_params.update({'alpha_h': alpha_h_target, 'omega_h': omega_h_target, 'delta_h': delta_h_target})


                    
                elif (mode == 'Distance'):
                    P_sol, solver_time = result_queue.get_nowait()  # This will not block


                if (solver_time is not None):
                    tot_solver_time += solver_time
                    num_mpc_solutions += 1

                waypoint_params = init_waypoint_parameters(P_sol.T)
                waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)
                prev_solution = P_sol.T

                if target_reached:
                    simulation_over = True
                

            except Empty:
                pass  # No new waypoints yet, continue with the current state
            

            # Simulation state update remains the same
            if mpc_thread is not None and not mpc_thread.is_alive():
                mpc_thread.join()  # Ensure thread resources are cleaned up if it's finished


            if (controller_params['transition_in_progress'] and t - controller_params['transition_start_time'] >= T):

                #print('finished transition')
                #Update controller params based on the retrieved solution
                controller_params.update({'alpha_h': alpha_h_target, 'omega_h': omega_h_target, 'delta_h': delta_h_target})
                controller_params.update({'transition_in_progress': False})
         


            # Integrate to get the next state
            if dimension == '3D':
                v = MX.sym('v', 4*n + 8)

            if dimension == '2D':
                v = MX.sym('v', 2*n + 7)

            v_dot, energy_consumption, phi_ref_x = calculate_v_dot_MPC(t, v, params, controller_params, waypoint_params, p_pathframe, dimension)
            energy_func = Function('energy_func', [v], [energy_consumption])
            phi_ref_x_func = Function('phi_ref_x_func', [v], [phi_ref_x])

            opts = {'tf': dt, 'abstol': 1e-4, 'reltol': 1e-4}
            F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)

            r = F(x0=v0)
            v0 = r['xf']  # Update the state vector for the next iteration
            t += dt  # Increment time

            energy = float(energy_func(v0))
            tot_energy += energy
            phi_ref_x_evaluated = phi_ref_x_func(v0)
            num_measurements += 1

            theta_x, theta_z, p_CM, theta_x_dot, theta_z_dot, p_CM_dot, y_int, z_int = extract_states(v0, n, dimension)

            distance_moved = np.linalg.norm(p_CM - p_CM_previous)
            total_distance_traveled += distance_moved
            p_CM_previous = np.copy(p_CM)

            p_CM_list.append(p_CM_previous.tolist())  # Assuming p_CM is a numpy array
            phi_ref_x_list.append(float(phi_ref_x_evaluated)) # Evaluate at v0)
            energy_list.append(energy)
            velocity_list.append(np.linalg.norm(p_CM_dot))
            theta_x_list.append(np.array(theta_x).tolist())
            theta_z_list.append(np.array(theta_z).tolist())


            if (not mpc_active and current_time >= next_mpc_update_time):
                    P = generate_initial_path(p_CM, target, k)
                    P = np.squeeze(P)
                    N = extend_horizon(P, N, obstacles, P.shape[0], controller_params)
                    waypoint_params = init_waypoint_parameters(P[:N,:])
                    next_mpc_update_time += mpc_dt  # Schedule next update
            
            
            waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)
                

            if target_reached:
                simulation_over = True
            
            x, y, z = draw_snake_robot(ax, target,  t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, dimension, None, alpha_h=None)
            #middle_link_list.append(middle_link_position.tolist())
            all_link_x.append(x)
            all_link_y.append(y)
            all_link_z.append(z)

            plt.pause(0.01)


    except Exception as e:
        print(f"Simulation failed due to: {e}")
        traceback.print_exc()  # Prints the detailed traceback

    plt.ioff()
    plt.close()


    avg_solver_time = tot_solver_time / num_mpc_solutions
    avg_power_usage = tot_energy / t

    log_data = {
    "p_CM_log": p_CM_list,
    "phi_ref_x": phi_ref_x_list,
    "velocity_list": velocity_list,
    "energy_list" : energy_list,
    "avg_solver_time" : avg_solver_time,
    "avg_power_usage" : avg_power_usage,
    "tot_energy" : tot_energy,
    "tot_distance" : total_distance_traveled,
    "tot_time"   : t,
    "theta_x_list": theta_x_list,
    "theta_z_list": theta_z_list,
    #"middle_link_log": middle_link_list,
    "all_link_x": all_link_x,
    "all_link_y": all_link_y,
    "all_link_z": all_link_z,
    "obstacles": [{"center": obs['center'], "radius": obs['radius']} for obs in obstacles],
     "target" : target
    }

    # Convert the entire data structure to a serializable format
    serializable_log_data = convert_to_serializable(log_data)

    log_file_path = "simulation_log.json"
    if not os.path.exists(log_file_path):
         with open(log_file_path, 'w') as file:
            json.dump({}, file)  # Or any other initial content

    with open(log_file_path, 'w') as file:
        json.dump(serializable_log_data, file, indent=4)


    return tot_energy, avg_power_usage, total_distance_traveled, t, avg_solver_time



def run_simulation_for_mode(mode, dimension):
    # Runs the simulation for the given mode and returns results
    tot_energy, avg_power_usage,  total_distance_traveled, t, avg_solver_time = start_simulation(mode, dimension)
    average_speed = total_distance_traveled / t
    return tot_energy,  avg_power_usage,  total_distance_traveled, average_speed, avg_solver_time

def print_results(mode, tot_energy, avg_power, total_distance, average_speed, avg_solver_time):
    # Print the results for the given mode
    print(f"\nResults for {mode} Mode:")
    print(f"Total energy: {tot_energy}")
    print(f"Avg power: {avg_power}")
    print(f"Total distance traveled: {total_distance} m")
    print(f"Average speed: {average_speed} m/s")
    print(f"Average MPC solver time: {avg_solver_time} s")


if __name__ == "__main__":

    modes = ['Distance']
    #modes = ['Energy']
    #modes = ['Energy_alpha']

    #dimension = '2D'
    dimension = '3D'

    results = {}

    for mode in modes:
        results[mode] = run_simulation_for_mode(mode, dimension)

    for mode in modes:
        print_results(mode, *results[mode])

    if dimension == '2D':
        visualize_simulation_results()
        plot_colored_path('2D')
        #plot_colored_path('2D', 'velocity')
        #plot_colored_path('2D', 'energy')
    
    if dimension == '3D':
        visualize_simulation_results_3d()
        #plot_colored_path('3D')
        plot_colored_path('3D', r_acceptance=0.5)

    

    


    









