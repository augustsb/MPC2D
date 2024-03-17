import numpy as np
import pandas as pd
from init_model_parameters import init_model_parameters
from init_controller_parameters import init_controller_parameters
from init_waypoint_parameters import init_waypoint_parameters
from calculate_pathframe_state import calculate_pathframe_state
from extract_states import extract_states
from calculate_v_dot_MPC import calculate_v_dot_MPC
from casadi import Function, integrator, vertcat, MX
import matplotlib.pyplot as plt
from draw_snake_robot import draw_snake_robot
from concurrent.futures import ProcessPoolExecutor
from predict_energy import load_and_preprocess_data
from filelock import Timeout, FileLock
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import traceback
import os
import random


def run_simulation(alpha_h, omega_h, delta_h):
    # Initialization
    params = init_model_parameters()
    n, l = params['n'], params['l']
    controller_params = init_controller_parameters(n, l)
    controller_params.update({'alpha_h': alpha_h, 'omega_h': omega_h, 'delta_h': delta_h})


    # Initial state setup
    theta_x0_dot, p_CM0, p_CM0_dot, y_int0 = params['theta_x0_dot'], params['p_CM0'], params['p_CM0_dot'], controller_params['y_int0']
    prev_theta_dot = theta_x0_dot
    theta_z = np.zeros(n)
    
    waypoints = np.array([[0, 0, 0], [2, 0, 0], [4, 2 ,0], [6, 2, 0], [8, 0, 0]])
    target = waypoints[-1,:]
    print(target)
    obstacles = [{'center': (2, 3, 0), 'radius': 1},]

    waypoint_params = init_waypoint_parameters(waypoints)
    waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM0, waypoint_params, controller_params, target)
    cur_alpha_path = waypoint_params['cur_alpha_path']
    theta_x0 = np.full(n, cur_alpha_path)
    v0 = vertcat(theta_x0, p_CM0, theta_x0_dot, p_CM0_dot, y_int0)


    start_time, stop_time, dt = 0, 400, 0.05
    t = start_time
    last_update_time = 0  # Last time the gait parameters were updated
    transition_period = 1.0
    increment_amount = 1*np.pi/180
    increment_interval = 2
    measurement_count = 0
    collecting_data = False  # Flag to indicate if we are in the data collection phase

    # Initialize placeholders for the start and end positions
    p_CM_start = np.array([0, 0, 0])
    measurement_count = 0
    total_energy = 0
    results = []

    #plt.close('all')
    #plt.ion()  # Turn on interactive plotting mode
    #fig, ax = plt.subplots()

    for t in np.arange(start_time, stop_time + dt, dt):
       # Update the parameters after every increment_interval + transition_period
        if t - last_update_time >= increment_interval + transition_period:
            if collecting_data:
                # Calculate the average velocity and energy for the previous set of parameters
                duration = t - (last_update_time + transition_period)
                p_CM_end = p_CM  # Assuming p_CM is updated in the simulation logic
                average_velocity = np.linalg.norm(p_CM_end - p_CM_start) / duration
                average_energy = total_energy / measurement_count
                
                # Store the results
                results.append({
                    'alpha_h': alpha_h,
                    'omega_h': omega_h,
                    'delta_h': delta_h,
                    'average_velocity': average_velocity,
                    'average_energy': average_energy
                })

                 # To print the latest values of 'alpha_h', 'omega_h', and 'delta_h'
                #print(f"Latest alpha_h: {results[-1]['alpha_h']}")
                #print(f"Latest omega_h: {results[-1]['omega_h']}")
                #print(f"Latest delta_h: {results[-1]['delta_h']}")
                #print(f"Latest average_velocity: {results[-1]['average_velocity']}")
                #print(f"Latest average_energy: {results[-1]['average_energy']}")
                
                # Reset for the next set of parameters
                collecting_data = False
                total_energy = 0
                measurement_count = 0
                if np.any(np.abs(p_CM_dot) > 5) or np.any(np.abs(theta_x_dot) > 20):
                    return results

            # Update gait parameters here
            alpha_h += increment_amount
            last_update_time = t
            p_CM_start = p_CM  # Reset start position for velocity calculation
            # Update controller parameters
            controller_params.update({'alpha_h': alpha_h, 'omega_h': omega_h, 'delta_h': delta_h})
            
        # Simulation logic...
        v = MX.sym('v', 2*n + 7)
        v_dot, energy_consumption = calculate_v_dot_MPC(t, v, params, controller_params, waypoint_params, p_pathframe, '2D')
        u_func = Function('u_func', [v], [energy_consumption])
        opts = {'tf': dt}
        F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
        r = F(x0=v0)
        v0 = r['xf']  # Update state vector
        t += dt  # Increment time

        theta_x, theta_z, p_CM, theta_x_dot, theta_z_dot, p_CM_dot, y_int, z_int = extract_states(v0, n, '2D')
        waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)

        if t >= last_update_time + transition_period:
            collecting_data = True
            energy_consumption = u_func(v0)
            # Accumulate energy and update p_CM_end for velocity calculation
            total_energy += float(energy_consumption.full())  # Converts DM to scalar floatenergy_consumption  # You need to calculate this in your simulation logic
            measurement_count += 1
        
        # Update the robot drawing
       # x, y, z = draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, '2D', None, alpha_h=None)
        #plt.pause(0.01)

     
    #plt.ioff()
    #plt.close()


    # Save results to a file
    #with open('simulation_results.json', 'w') as file:
       # json.dump(results, file, indent=4)

    return results




def safe_write_results_to_file(results, filename='simulation_results.json'):
    lock = FileLock(f"{filename}.lock")
    with lock:
        try:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []
        
        # Add the new results to the existing data
        if isinstance(results, dict):
            existing_data.append(results)  # If results is a single dict, append it
        else:
            existing_data.extend(results)  # If results is a list, extend the list
        
        # Write the updated data back to the file
        with open(filename, 'w') as file:
            json.dump(existing_data, file, indent=4)



def generate_parameter_sets_for_omega(alpha_start, omega_start, omega_end, omega_increment, delta_h):
    """Generate parameter sets, varying omega_h while keeping others fixed initially."""
    omega_h_values = np.arange(omega_start, omega_end + omega_increment, omega_increment)
    return [(alpha_start, omega_h, delta_h) for omega_h in omega_h_values]



def run_simulation_wrapper(params):
    results = run_simulation(*params)
    safe_write_results_to_file([results])  # Assuming results is a dict; wrap it in a list if needed

def main():
    alpha_h_start, omega_h_start, delta_h_start = 0, 0, np.radians(40)
    omega_h_end = np.radians(210)
    omega_increment = np.radians(1)
    
    parameter_sets = generate_parameter_sets_for_omega(alpha_h_start, omega_h_start, omega_h_end, omega_increment, delta_h_start)
    
    with ProcessPoolExecutor() as executor:
        # Using list comprehension to directly submit and process simulations
        [executor.submit(run_simulation_wrapper, params) for params in parameter_sets]

# This ensures the script can be run as a standalone program
if __name__ == "__main__":
    main()


# Example usage
alpha_h_start, omega_h_start, delta_h_start = 0 * np.pi / 180, 150 * np.pi / 180, 40 * np.pi / 180
results = run_simulation(alpha_h_start, omega_h_start, delta_h_start)
print(results)

