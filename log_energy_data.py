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
from filelock import Timeout, FileLock
import traceback
import os


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
    
    waypoints = np.array([[0, 0, 0], [3, 3, 0]])
    target = [3, 3, 0]
    obstacles = [{'center': (2, 3, 0), 'radius': 1},]

    waypoint_params = init_waypoint_parameters(waypoints)
    waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM0, waypoint_params, controller_params, target)
    cur_alpha_path = waypoint_params['cur_alpha_path']
    theta_x0 = np.full(n, cur_alpha_path)
    v0 = vertcat(theta_x0, p_CM0, theta_x0_dot, p_CM0_dot, y_int0)

    # Simulation loop

    tot_energy = 0
    start_time, stop_time, dt = 0, 30,  0.05
    t = start_time
    current_time = 0  # Initialize current time
    warm_up_time = 2  # Warm-up time in seconds
    collect_time = 4
    #velocities = []
    #early_stop_time = 0.5
    in_warm_up = True  # Flag to indicate if in warm-up phase
    p_CM_post_warm_up = np.zeros(3)
    #num_timesteps = 0
    simulation_over = 0
    success = False
    effective_simulation_duration = 0  # Initialize here

    #plt.close('all')
    #plt.ion()  # Turn on interactive plotting mode
    #fig, ax = plt.subplots()

    try:
        while not simulation_over and t < stop_time:
            v = MX.sym('v', 2*n + 7)
            v_dot, energy_consumption = calculate_v_dot_MPC(t, v, params, controller_params, waypoint_params, p_pathframe)
            u_func = Function('u_func', [v], [energy_consumption])
            opts = {'tf': dt}
            F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
            r = F(x0=v0)
            v0 = r['xf']  # Update state vector
            t += dt  # Increment time
            current_time += dt  # Update current time
            
            if in_warm_up and current_time > warm_up_time:
                # Capture the position of CM right after warm-up phase ends
                p_CM_dot = np.array(p_CM_dot)
                if np.array_equal(p_CM_dot, np.zeros(3)):
                       success = False
                       simulation_over = True
                       tot_energy = 0
                in_warm_up = False  # Mark end of warm-up phase

            theta_x, p_CM, theta_x_dot, p_CM_dot, y_int = extract_states(v0, n)
            waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)


            if np.any(np.abs(p_CM_dot) > 5) or np.any(np.abs(theta_x_dot) > 5):
                success = False
                simulation_over = True
 
            if  t >= collect_time:

                if (np.array_equal(p_CM_post_warm_up, np.zeros(3))):
                    p_CM_post_warm_up = np.copy(p_CM)  
                #velocities.append(np.abs(p_CM_dot))
                energy = u_func(v0)*dt
                tot_energy += energy

            if target_reached:
                 success = True
                 simulation_over = True

            #draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, alpha_h)
            #plt.pause(0.01)

        effective_simulation_duration = t - collect_time

        #plt.ioff()

        #plt.close()


    except Exception as e:
            print(f"Simulation failed due to: {e}")
            traceback.print_exc()  # Prints the detailed traceback
            tot_energy = 0
            success = False  # Mark simulation as unsuccessful
            simulation_over


    # Compile simulation results
    try:
        if effective_simulation_duration > 0:  # Ensure no division by zero
            simulation_results = {
                'average_velocity': np.linalg.norm(p_CM - p_CM_post_warm_up) / effective_simulation_duration,
                'average_energy': tot_energy / effective_simulation_duration,
                'success': success
            }
        else:
            simulation_results = {
                'average_velocity': 0,
                'average_energy': 0,
                'success': success
            }
        return simulation_results
    
    except Exception as e:
            print(f"Error compiling simulation results: {e}")
            simulation_results = {
                    'average_velocity': 0,
                    'average_energy': 0,
                    'success': success
                }
            return simulation_results




def run_and_save_simulation(alpha_h, omega_h, delta_h, filename="results.csv"):
    simulation_results = run_simulation(alpha_h, omega_h, delta_h)
    
    # Prepare data to be saved
    data_to_save = {
        'alpha_h': alpha_h,
        'omega_h': omega_h,
        'delta_h': delta_h,
        **simulation_results
    }
    
    # Append results to the CSV file
    # Ensure thread-safe write access
    lock_path = filename + ".lock"
    with FileLock(lock_path):
        
        # Append data to the CSV file
        df_to_append = pd.DataFrame([data_to_save])
        if not os.path.exists(filename):
            df_to_append.to_csv(filename, mode='w', header=True, index=False)
        else:
            df_to_append.to_csv(filename, mode='a', header=False, index=False)
        

def main():
    # Define the step size in degrees and convert to radians, etc.
    step_size_deg = 5
    step_size_rad = step_size_deg * np.pi / 180
  
    alpha_h_values = np.arange(0, 90 + step_size_deg, step_size_deg) * np.pi / 180
    omega_h_values = np.arange(0, 210 + step_size_deg, step_size_deg) * np.pi / 180
    delta_h_values = np.arange(0, 90 + step_size_deg, step_size_deg) * np.pi / 180

    filename = "parallel_results.csv"
    num_processes = os.cpu_count()  # Adjust as needed

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for alpha_h in alpha_h_values:
            for omega_h in omega_h_values:
                for delta_h in delta_h_values:
                    future = executor.submit(run_and_save_simulation, alpha_h, omega_h, delta_h, filename)
                    futures.append(future)

        for future in futures:
            future.result()  # Wait for all submitted tasks to complete

    print("All simulations completed and results saved.")

if __name__ == '__main__':
    main()





"""
# Initialize an empty DataFrame to store results
columns = ['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'average_energy', 'success']
results_df = pd.DataFrame(columns=columns)

# Define the step size in degrees
step_size_deg = 10

# Convert the step size to radians
step_size_rad = step_size_deg * np.pi / 180


# Create the ranges for each parameter in radians
#alpha_h_range = np.arange(30, 90 + step_size_deg, step_size_deg) * np.pi / 180
#omega_h_range = np.arange(150, 210 + step_size_deg, step_size_deg) * np.pi / 180
#delta_h_range = np.arange(40, 90 + step_size_deg, step_size_deg) * np.pi / 180


alpha_h_range = np.array([30*np.pi/180])
omega_h_range = np.array([150*np.pi/180]) 
delta_h_range = np.array([40*np.pi/180])


#0.17453292519943295,0.3490658503988659,0.17453292519943295

# Simulation counter and save interval
simulation_counter = 0
save_interval = 50  # Save after every 100 simulations, for example

# Loop through your parameters
for alpha_h in alpha_h_range:
    for omega_h in omega_h_range:
        for delta_h in delta_h_range:
            # Run your simulation
            simulation_results = run_simulation(alpha_h, omega_h, delta_h)
            # Append results to the DataFrame
            results_df = results_df.append({
                'alpha_h': alpha_h,
                'omega_h': omega_h,
                'delta_h': delta_h,
                **simulation_results  # Unpack simulation results directly into the DataFrame row
            }, ignore_index=True)
            
            # Increment simulation counter
            simulation_counter += 1
            
            # Save periodically
            if simulation_counter % save_interval == 0:
                results_df.to_csv('results.csv', index=False)
                print(f"Saved results after {simulation_counter} simulations.")

# Final save to ensure all results are stored
results_df.to_csv('results.csv', index=False)
print("All simulation results saved.")



"""




