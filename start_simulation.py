from casadi import MX,  vertcat, integrator, Opti
import matplotlib.pyplot as plt
from init_model_parameters import init_model_parameters
from init_controller_parameters import init_controller_parameters
from calculate_v_dot import calculate_v_dot
from draw_snake_robot import draw_snake_robot
from extract_states import extract_states
from init_waypoint_parameters import init_waypoint_parameters
from calculate_pathframe_state import calculate_pathframe_state
import numpy as np


def start_simulation():
    # Initialize parameters
    params = init_model_parameters()
    
    n = params['n']
    l = params['l']

    controller_params = init_controller_parameters(n,l)
    waypoint_params = init_waypoint_parameters()
    

    start_time = 0
    stop_time = 200
    dt = 0.05


    theta_x0 = params['theta_x0']
    theta_x0_dot = params['theta_x0_dot']
    p_CM0 = params['p_CM0']
    p_CM0_dot = params['p_CM0_dot'] 
    y_int0 = controller_params['y_int0']

    theta_z = np.zeros(n)

    # Define obstacles
    obstacles = [{
        'center': [2, 0, 0],  # Center of the obstacle
        'radius': 0.2         # Radius of the obstacle
    }]



    waypoint_params, p_pathframe = calculate_pathframe_state(p_CM0, waypoint_params)
    # Initial state vector
    v0 = vertcat(theta_x0, p_CM0, theta_x0_dot, p_CM0_dot, y_int0)  # Define these variables


    # Prepare for real-time plotting
    plt.ion()  # Turn on interactive plotting mode
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    t = start_time  # Initialize time
    while t < stop_time:
        

        v = MX.sym('v', 2*n + 5)
        v_dot = calculate_v_dot(t, v, params, controller_params, waypoint_params, p_pathframe)
        opts = {'tf': dt}
        F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
        #opts = {'tf': dt, 'linear_solver': 'csparse', 'max_num_steps': 10000}  # Example options for stiff problems
        #F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
        r = F(x0=v0)
        v0 = r['xf']  # Update the state vector for the next iteration
        t += dt  # Increment time

        # Extract states and visualize or store data as needed...
        theta_x, p_CM, theta_x_dot, p_CM_dot, y_int = extract_states(v0, n)

        waypoint_params, p_pathframe = calculate_pathframe_state(p_CM, waypoint_params)


        draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles)  # Example call
        plt.pause(0.05)
      

    plt.ioff()
    plt.show()

    

# Call the start_simulation function to begin the simulation
if __name__ == "__main__":
    start_simulation()

