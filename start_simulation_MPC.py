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
import csv
import traceback


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
    num_obstacles = 8
    area_size = (31, 8)
    min_distance_to_start_target = 2.0
    #obstacles = generate_random_obstacles(num_obstacles, area_size, p_CM, target, min_distance_to_start_target)

    #obstacles = [{'center': (10, 0, 0), 'radius': 1.5},]

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

    """



    dt = 0.05  #Update frequency simulation
    mpc_dt = 1 #Update frequency mpc
    k = 1 #desired step length
    initial_N = 10 # Initial prediction horizon
    N = initial_N


    P, num_states = generate_initial_path(p_CM, target, k)
    P = np.squeeze(P)
    #if (P_sol is None):
       #P_sol = P.T
    #P, num_states = path_resolution(P, P.T, k, N)
    N = extend_horizon(P, N, obstacles, num_states, controller_params)


    result_queue = Queue()

    if (mode == 'Distance'):
        mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM, target, obstacles, params, controller_params, N, k, result_queue, P))
        mpc_thread.start()
        try:
            P_sol = result_queue.get()  # This will block until a solution is available
        except:
            print("Failed to generate initial waypoints")

    
    if (mode == 'Energy'):
        mpc_thread = threading.Thread(target=mpc_energy_efficiency, args=(p_CM, p_CM_dot, target, obstacles, params, controller_params, N, k, result_queue, P))
        mpc_thread.start()
        
        try:
            P_sol,  alpha_h_sol, omega_h_sol, delta_h_sol,  V_sol = result_queue.get()  # This will block until a solution is available
            print('alpha_h:', alpha_h_sol[0])
            print('omega_h:', omega_h_sol[0])
            controller_params.update({'alpha_h': alpha_h_sol[0], 'omega_h': omega_h_sol[0], 'delta_h': delta_h_sol[0]})
        except:
            print("Failed to generate initial waypoints")


    waypoint_params = init_waypoint_parameters(P_sol.T)
    waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)
    cur_alpha_path = waypoint_params['cur_alpha_path']
    theta_x0 = np.full(n, cur_alpha_path)

     
    v0 = vertcat(theta_x0, p_CM, theta_x0_dot, p_CM_dot, y_int)

    simulation_over = 0
    total_distance_traveled = 0
    t = 0
    p_CM_previous = p_CM
    tot_energy = 0
    next_mpc_update_time = mpc_dt
    result_queue = Queue()
    mpc_thread = None

    plt.ion()  # Turn on interactive plotting mode
    fig, ax = plt.subplots()

    try: 
        while not simulation_over:

            current_time = t
            # Check if it's time to start or update the MPC calculation
            if current_time >= next_mpc_update_time:

                if mpc_thread is None or not mpc_thread.is_alive():

                    P, num_states = generate_initial_path(p_CM, target, k)
                    P = np.squeeze(P)
                    if (P_sol is None):
                        P_sol = P.T
                    P, num_states = path_resolution(P, P_sol, k, N)
                    N = extend_horizon(P, N, obstacles, num_states, controller_params)

                    if (mode == 'Energy'):
                        mpc_thread = threading.Thread(target=mpc_energy_efficiency(p_CM, p_CM_dot,  target, obstacles, params, controller_params,  N, k, result_queue, P))

                    elif (mode == 'Distance'):
                        mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM,  target, obstacles, params, controller_params, N, k,  result_queue, P))

                    mpc_thread.start()
                    next_mpc_update_time += mpc_dt  # Schedule next update
                    
            # Try to get the MPC results without blocking
            try:

                if (mode == 'Energy'):
                    P_sol,  alpha_h_sol, omega_h_sol, delta_h_sol,  V_sol = result_queue.get_nowait()
                    print('alpha_h:', alpha_h_sol[0])
                    #print('omega_h:', omega_h_sol[0])
                    #print(V_sol)
                    controller_params.update({'alpha_h': alpha_h_sol[0], 'omega_h': omega_h_sol[0], 'delta_h': delta_h_sol[0]})
                    

                elif (mode == 'Distance'):

                    P_sol = result_queue.get_nowait()  # This will not block

                waypoint_params = init_waypoint_parameters(P_sol.T)
                waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

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
    
            waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

            if target_reached:
                simulation_over = True
            
            draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, None, alpha_h=None)
            plt.pause(0.01)


    except Exception as e:
        print(f"Simulation failed due to: {e}")
        traceback.print_exc()  # Prints the detailed traceback

    plt.ioff()
    plt.close()

    return tot_energy, total_distance_traveled, t



def run_simulation_for_mode(mode):
    # Runs the simulation for the given mode and returns results
    tot_energy, total_distance, t = start_simulation(mode)
    average_speed = total_distance / t
    return tot_energy, total_distance, average_speed

def print_results(mode, tot_energy, total_distance, average_speed):
    # Print the results for the given mode
    print(f"\nResults for {mode} Mode:")
    print(f"Total energy: {tot_energy}")
    print(f"Total distance traveled: {total_distance} units")
    print(f"Average speed: {average_speed} units/time")


if __name__ == "__main__":
    modes = ['Distance', 'Energy']
    results = {}

    for mode in modes:
        results[mode] = run_simulation_for_mode(mode)

    # Now, results contain the simulation outcomes for both modes
    # You can compare or print them as needed
    for mode in modes:
        print_results(mode, *results[mode])


    









