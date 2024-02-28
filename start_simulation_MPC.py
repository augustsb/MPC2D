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
import threading
from queue import Queue, Empty  # Note the import of Empty here
from waypoint_methods import generate_initial_waypoints, calculate_curvature, calculate_distances, calculate_turning_angles
import csv
import traceback

def format_data(simulation_results):
    # Convert CasADi DM objects to float for formatting
    formatted_results = {
        'tot_energy': "{:.2f}".format(float(simulation_results['tot_energy'])),
        'total_distance': "{:.2f}".format(float(simulation_results['total_distance'])),
        'total_curvature': "{:.2f}".format(float(simulation_results['total_curvature'])),
        'max_angle': "{:.2f}".format(float(simulation_results['max_angle'])),
        'success': "Yes" if simulation_results['success'] else "No"
    }
    return formatted_results



def start_simulation():
    # Initialize parameters

        # Initialize variables with default values
    tot_energy = 0
    total_distance = 0
    total_curvature = 0
    success = False  # Assuming success is a boolean
    max_angle = 0
    num_timesteps = 0
    tot_power = 0

    params = init_model_parameters()

    n = params['n']
    l = params['l']

    controller_params = init_controller_parameters(n,l)
    y_int = controller_params['y_int0']

    theta_z = np.zeros(n)
    #theta_x0 = params['theta_x0']
    theta_x0_dot = params['theta_x0_dot']
    p_CM = params['p_CM0']
    p_CM_dot = params['p_CM0_dot'] 

    key_states = []
    num_obstacles = 8  # Number of obstacles to generate
    area_size = (16, 8)  # Size of the area
    min_distance_to_target = 2
    width, height = area_size
    min_x, min_y = 8, 6  # Define min bounds if any, assuming (0, 0) for simplicity
    target = np.array([5, 5, 0])
    # Generate random target within bounds
    #target_x = np.random.uniform(min_x, width)
    #target_y = np.random.uniform(min_y, height)
    #target_z = 0  # Assuming a 2D plane for simplicity, set Z to 0 if it's 3D but flat

    #target = np.array([target_x, target_y, target_z])
   
    start = p_CM


    #obstacles = generate_random_obstacles(num_obstacles, area_size, start, target, min_distance_to_target)

   
    obstacles = [{'center': (2, 3, 0), 'radius': 1},]
    
    """
    waypoints = np.array([[0, 0, 0], [1,  0 , 0]
                        ,[2, 0, 0]
                        ,[3, 0 ,0]
                         ,[4.51775321e+00, 1.47724675e+00,0]
                         ,[4.83336555e+00, 2.42613496e+00,0]
                        ,[4.91185474e+00, 3.42304988e+00,0]
                        ,[4.96766418e+00, 4.42149117e+00,0]
                         ,[5.57089833e+00, 4.71511850e+00,0]
                         ,[5.00000000e+00, 5.00000000e+00,0]])

    """

    dt = 0.05  #Update frequency simulation
    mpc_dt = 1
    k = 1
    N = 10

    #waypoints = generate_initial_waypoints(start, target, N)
    waypoints = np.array([[0, 0, 0], [2, 2, 0]])

    result_queue = Queue()
    mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM, target, obstacles, params, controller_params, waypoints,
    N, k, result_queue))
    mpc_thread.start()
    #waypoints, alpha_h = mpc_shortest_path(p_CM, p_CM_dot, target, obstacles, params, controller_params, waypoints, N, k, result_queue=None)


    #waypoints, alpha_h = result_queue.get()  # This will block until a solution is available
    #new_waypoints = result_queue.get()  # This will block until a solution is available

    #angles, curvature, max_angle = calculate_turning_angles(waypoints)
    #distances, total_distance = calculate_distances(waypoints)
    #total_curvature = curvature / total_distance


    waypoint_params = init_waypoint_parameters(waypoints)
    waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

    cur_alpha_path = waypoint_params['cur_alpha_path']
    theta_x0 = np.full(n, cur_alpha_path)
    #theta_x0 = params['theta_x0']
     
    v0 = vertcat(theta_x0, p_CM, theta_x0_dot, p_CM_dot, y_int)

    plt.ion()  # Turn on interactive plotting mode
    fig, ax = plt.subplots()

    #while t < stop_time:
    simulation_over = 0
    t = 0
    next_mpc_update_time = mpc_dt
    result_queue = Queue()
    mpc_thread = None

    default_alpha_h_value = 30 * np.pi / 180  # Example: 45 degrees in radians as a default
    alpha_h = np.full((1, N+1), default_alpha_h_value)  # Default alpha_h for all waypoints

    try: 
        while not simulation_over:
            current_time = t

            """
      
            # Check if it's time to start or update the MPC calculation
            if current_time >= next_mpc_update_time:
                if mpc_thread is None or not mpc_thread.is_alive():
                    # Start or restart the MPC calculation
                    initial_waypoints = generate_initial_waypoints(p_CM, target, N)
                    mpc_thread = threading.Thread(target=mpc_shortest_path, args=(p_CM,  target, obstacles, params, controller_params, initial_waypoints,
                                                                                N, k, result_queue))
                    mpc_thread.start()
                    next_mpc_update_time += mpc_dt  # Schedule next update
                    
            # Try to get the MPC results without blocking
            try:
                #waypoints, alpha_h = result_queue.get_nowait()  # This will not block
                new_waypoints = result_queue.get_nowait()  # This will not block
                waypoint_params = init_waypoint_parameters(new_waypoints)
                waypoint_params, p_pathframe, target_reached  = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

            except Empty:
                pass  # No new waypoints yet, continue with the current state
            
            # Simulation state update remains the same

            if mpc_thread is not None and not mpc_thread.is_alive():
                mpc_thread.join()  # Ensure thread resources are cleaned up if it's finished

           """
        
    
            # Integrate to get the next state
            v = MX.sym('v', 2*n + 7)
            v_dot, energy_consumption = calculate_v_dot_MPC(t, v, params, controller_params, waypoint_params, p_pathframe)
            energy_func = Function('energy_func', [v], [energy_consumption])
            opts = {'tf': dt}
            F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)

            r = F(x0=v0)
            v0 = r['xf']  # Update the state vector for the next iteration
            t += dt  # Increment time


            """
            num_timesteps += 1
            energy_val = energy_func(v0)
            tot_energy += energy_val
            tot_power += energy_val / dt
            """
 
            theta_x, p_CM, theta_x_dot, p_CM_dot, y_int = extract_states(v0, n)
            print(p_CM_dot)
            print(theta_x_dot)


            #Snake goes wild
            if np.any(np.abs(theta_x_dot) > 20):
                tot_energy += 1e12
                success = False
                simulation_over = True


            waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

            #Reached target
            if target_reached:
                success = True
                simulation_over = True
            
            draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, alpha_h)
            plt.pause(0.01)


 
    except Exception as e:
        print(f"Simulation failed due to: {e}")
        traceback.print_exc()  # Prints the detailed traceback
        tot_energy += 1e12
        success = False  # Mark simulation as unsuccessful


    # Simulation results
    simulation_results = {
        'tot_energy': tot_energy,
        'total_distance': total_distance,
        'total_curvature': total_curvature,
        'success': success,
        'max_angle': max_angle,  # Include this in the results
        'num_timeteps' : num_timesteps,
        'tot_power' : tot_power, 
    }


    plt.ioff()

    plt.close()


    return simulation_results




if __name__ == "__main__":

    # Define the path to your CSV file
    data_file_path = 'simulation_results.csv'

    # Open the file for writing. Use mode 'a' to append or 'w' to overwrite
    with open(data_file_path, mode='a', newline='') as file:
        data_writer = csv.writer(file)
        
        # Write headers if the file is new/empty or if you're starting fresh
        if file.tell() == 0:  # Checks if the file is empty
            data_writer.writerow(['Run', 'Total Energy', 'Total Distance', 'Total Curvature', 'Max angle', 'Success'])

        # Run the simulation 5 times
        for run in range(1):
            simulation_results = start_simulation()

                    # Format the results for readability
            formatted_results = format_data(simulation_results)

            # Write the formatted results of this run to the CSV file
            data_writer.writerow([
                run,
                formatted_results['tot_energy'],
                formatted_results['total_distance'],
                formatted_results['total_curvature'],
                formatted_results['max_angle'],
                formatted_results['success']
            ])









