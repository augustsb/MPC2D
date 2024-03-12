from casadi import MX,  vertcat, integrator, Opti, Function
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from init_controller_parameters import init_controller_parameters
from calculate_v_dot_MPC import calculate_v_dot_MPC
from draw_snake_robot import draw_snake_robot
from extract_states import extract_states
from init_waypoint_parameters import init_waypoint_parameters
from calculate_pathframe_state import calculate_pathframe_state
import numpy as np
from generate_random_obstacles import generate_random_obstacles
from MPC_shortest_path import mpc_shortest_path
from MPC_energy_efficiency import mpc_energy_efficiency
import threading
from queue import Queue, Empty  # Note the import of Empty here
from waypoint_methods import  generate_initial_path, path_resolution, extend_horizon
import json
import traceback
from improved_initial_guess import rrt, interpolate_path
from visualize_simulation_results import visualize_simulation_results, convert_to_serializable
from predict_energy import load_and_preprocess_data


data_all =  load_and_preprocess_data("/home/augustsb/MPC2D/results_2802", "chunk_results_", 16)



def start_simulation(mode):
    # Initialize parameters

    params = init_model_parameters()

    n = params['n']
    l = params['l']

    controller_params = init_controller_parameters(n,l)

    y_int = controller_params['y_int0']
    theta_z = np.zeros(n)
    theta_x0_dot = params['theta_x0_dot']
    p_CM = params['p_CM0']
    p_CM_dot = params['p_CM0_dot'] 

    
    target = np.array([29.0 , 0.0, 0.0])
    #target = np.array([3.0 , 0.0, 0.0])
    #num_obstacles = 8
    #rea_size = (31, 8)
    #min_distance_to_start_target = 2.0
    #obstacles = generate_random_obstacles(num_obstacles, area_size, p_CM, target, min_distance_to_start_target)

    #obstacles = [{'center': (10, 0, 0), 'radius': 1.5},]
    """
    obstacles = [
        {'center': (10, 1.8, 0), 'radius': 1.5},  # First obstacle
        {'center': (10, -1.8, 0), 'radius': 1.5}, # Second obstacle
    ]
    """
 
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


    dt = 0.05  #Update frequency simulation
    mpc_dt = 1 #Update frequency mpc
    k = 1 #desired step length
    initial_N = 10 # Initial prediction horizon
    N = initial_N
    N_min = 2
    V_min = 0.2

    
    all_params = True
    simulation_over = False
    total_distance_traveled = 0
    t = 0
    p_CM_previous = p_CM
    p_CM_list = []
    middle_link_list = []
    alpha_h_list = []
    tot_energy = 0
    tot_solver_time = 0
    num_mpc_solutions = 0
    next_mpc_update_time = mpc_dt
 
  
    P = generate_initial_path(p_CM, target, k)
    P = np.squeeze(P)
    if (N > P.shape[0] - 1): 
         N = P.shape[0] - 1
    initial_N = N
    N = extend_horizon(P, N, obstacles, P.shape[0], controller_params)

    """
    start_point = tuple(p_CM.tolist())  # Convert p_CM (if it's a numpy array) to tuple
    goal_point = tuple(P[N-1, :].tolist())  # Convert P[N-1, :] to tuple
    P = rrt(start_point, goal_point, obstacles, controller_params['alpha_h'])
    P = interpolate_path(P, N)
    """

    result_queue = Queue()

    if (mode == 'Distance'):
        mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM, target, obstacles, params, controller_params,
                                                                       N, k, result_queue, P))
        mpc_thread.start()
        try:
            P_sol, solver_time = result_queue.get()  # This will block until a solution is available
            tot_solver_time += solver_time
            num_mpc_solutions += 1
        except:
            print("Failed to generate initial waypoints")

    
    if (mode == 'Energy'):
        mpc_thread = threading.Thread(target=mpc_energy_efficiency, args=(p_CM, p_CM_dot, target, obstacles, params,
                                                                           controller_params, N, k, result_queue, P, all_params))
        mpc_thread.start()
        
        try:
            result = result_queue.get()  # This will block until a solution is available

            # Retrieve common results
            P_sol = result["sol_waypoints"]
            sol_alpha_h = result["sol_alpha_h"]
            solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set

            # Update controller params based on the retrieved solution
            controller_params.update({'alpha_h': sol_alpha_h[0]})  # Example for alpha_h

            #valid_entries = data_all[(data_all['alpha_h'] == sol_alpha_h[0]) & (data_all['average_velocity'] >= V_min)]
            #optimal_entry = valid_entries.loc[valid_entries['average_energy'].idxmin()]
            #controller_params.update({'omega_h': optimal_entry['omega_h']})
            #controller_params.update({'delta_h': optimal_entry['delta_h']})



            if "sol_omega_h" in result and "sol_delta_h" in result:
                # These values are only present if all_params was True
                sol_omega_h = result["sol_omega_h"]
                sol_delta_h = result["sol_delta_h"]
                sol_V = result["sol_V"]
                controller_params.update({'omega_h': sol_omega_h[0], 'delta_h': sol_delta_h[0]})

            # Aggregate solver time and count
            tot_solver_time += solver_time
            num_mpc_solutions += 1
        except:
            print("Failed to generate initial waypoints")

    
    waypoint_params = init_waypoint_parameters(P_sol.T)
    waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)
    cur_alpha_path = waypoint_params['cur_alpha_path']
    theta_x0 = np.full(n, cur_alpha_path)

    v0 = vertcat(theta_x0, p_CM, theta_x0_dot, p_CM_dot, y_int)


    plt.ion()  # Turn on interactive plotting mode
    fig, ax = plt.subplots()

    result_queue = Queue()
    mpc_thread = None
    
    try: 
        while not simulation_over:
            
            current_time = t
            # Check if it's time to start or update the MPC calculation
            if (current_time >= next_mpc_update_time and N > N_min):
             
                if mpc_thread is None or not mpc_thread.is_alive():
                
                    P = generate_initial_path(p_CM, target, k)
                    P = np.squeeze(P)
                    N = extend_horizon(P, N, obstacles, P.shape[0], controller_params)
                    
                    """
                    start_point = tuple(np.array(p_CM.full()).flatten())
                    goal_point = tuple(P[N-1, :].tolist())  # Convert P[N-1, :] to tuple
                    P = rrt(start_point, goal_point, obstacles, controller_params['alpha_h'])
                    P = interpolate_path(P, N)
                    """

                    if (mode == 'Energy'):
                        mpc_thread = threading.Thread(target=mpc_energy_efficiency(p_CM, p_CM_dot,  target, obstacles, params, controller_params,
                                                                                     N, k, result_queue, P, all_params))

                    elif (mode == 'Distance'):
                        mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM,  target, obstacles, params, controller_params,
                                                                                       N, k,  result_queue, P))

                    mpc_thread.start()
                    next_mpc_update_time += mpc_dt  # Schedule next update
                    
            # Try to get the MPC results without blocking
            try:
                
                if (mode == 'Energy'):
   
                    result = result_queue.get_nowait()  # This will block until a solution is available
 
                    P_sol = result["sol_waypoints"]
                    sol_alpha_h = result["sol_alpha_h"]
                    solver_time = result.get("solver_time", 0)  # Use .get to provide a default value in case it's not set
                    # Update controller params based on the retrieved solution
                    controller_params.update({'alpha_h': sol_alpha_h[0]})  # Example for alpha_h

                    #valid_entries = data_all[(data_all['alpha_h'] == sol_alpha_h[0]) & (data_all['average_velocity'] >= V_min)]
                    #optimal_entry = valid_entries.loc[valid_entries['average_energy'].idxmin()]
                    #controller_params.update({'omega_h': optimal_entry['omega_h']})
                    #controller_params.update({'delta_h': optimal_entry['delta_h']})


                    if "sol_omega_h" in result and "sol_delta_h" in result:
                        # These values are only present if all_params was True
                        sol_omega_h = result["sol_omega_h"]
                        sol_delta_h = result["sol_delta_h"]
                        controller_params.update({'omega_h': sol_omega_h[0], 'delta_h': sol_delta_h[0]})
                    
                elif (mode == 'Distance'):
                    P_sol, solver_time = result_queue.get_nowait()  # This will not block


                if (solver_time is not None):
                    tot_solver_time += solver_time
                    num_mpc_solutions += 1

                waypoint_params = init_waypoint_parameters(P_sol.T)
                waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

                if target_reached:
                    simulation_over = True
                

            except Empty:
                pass  # No new waypoints yet, continue with the current state
            

            # Simulation state update remains the same
            if mpc_thread is not None and not mpc_thread.is_alive():
                mpc_thread.join()  # Ensure thread resources are cleaned up if it's finished

    
            # Integrate to get the next state
            v = MX.sym('v', 2*n + 7)
            v_dot, energy_consumption = calculate_v_dot_MPC(t, v, params, controller_params, waypoint_params, p_pathframe)
            energy_func = Function('energy_func', [v], [energy_consumption])
            opts = {'tf': dt, 'abstol': 1e-4, 'reltol': 1e-4}
            F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)

            r = F(x0=v0)
            v0 = r['xf']  # Update the state vector for the next iteration
            t += dt  # Increment time
            tot_energy += energy_func(v0)*dt

            theta_x, p_CM, theta_x_dot, p_CM_dot, y_int = extract_states(v0, n)

            distance_moved = np.linalg.norm(p_CM - p_CM_previous)
            total_distance_traveled += distance_moved
            p_CM_previous = np.copy(p_CM)
            p_CM_list.append(p_CM_previous.tolist())  # Assuming p_CM is a numpy array

            waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

            if target_reached:
                simulation_over = True
            
            middle_link_position = draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, None, alpha_h=None)
            middle_link_list.append(middle_link_position.tolist())

            plt.pause(0.01)


    except Exception as e:
        print(f"Simulation failed due to: {e}")
        traceback.print_exc()  # Prints the detailed traceback

    plt.ioff()
    plt.close()


    log_data = {
    "p_CM_log": p_CM_list,
    "middle_link_log": middle_link_list,
    "obstacles": [{"center": obs['center'], "radius": obs['radius']} for obs in obstacles]
    }

    # Convert the entire data structure to a serializable format
    serializable_log_data = convert_to_serializable(log_data)

    log_file_path = "simulation_log.json"
    with open(log_file_path, 'w') as file:
        json.dump(serializable_log_data, file, indent=4)



    avg_solver_time = tot_solver_time / num_mpc_solutions

    return tot_energy, total_distance_traveled, t, avg_solver_time



def run_simulation_for_mode(mode):
    # Runs the simulation for the given mode and returns results
    tot_energy, total_distance, t, avg_solver_time = start_simulation(mode)
    average_speed = total_distance / t
    return tot_energy, total_distance, average_speed, avg_solver_time

def print_results(mode, tot_energy, total_distance, average_speed, avg_solver_time):
    # Print the results for the given mode
    print(f"\nResults for {mode} Mode:")
    print(f"Total energy: {tot_energy}")
    print(f"Total distance traveled: {total_distance} m")
    print(f"Average speed: {average_speed} m/s")
    print(f"Average MPC solver time: {avg_solver_time} s")


if __name__ == "__main__":
    #modes = ['Distance', 'Energy']
    #modes = ['Distance']
    modes = ['Energy']
    results = {}

    for mode in modes:
        results[mode] = run_simulation_for_mode(mode)

    for mode in modes:
        print_results(mode, *results[mode])

    visualize_simulation_results()

    


    









