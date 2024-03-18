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

x_list = []
y_list = []

def start_simulation():
    # Initialize parameters
    params = init_model_parameters()
    
    n = params['n']
    l = params['l']


    waypoints = np.array([[0,0,0], [2,0,0]])

    controller_params = init_controller_parameters(n,l)
    waypoint_params = init_waypoint_parameters(waypoints)



    start_time = 0
    stop_time = 10
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

    target = np.array([2,0,0])



    waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM0, waypoint_params, controller_params, target)
    # Initial state vector
    v0 = vertcat(theta_x0, p_CM0, theta_x0_dot, p_CM0_dot, y_int0)  # Define these variables


    # Prepare for real-time plotting
    plt.ion()  # Turn on interactive plotting mode
    fig, ax = plt.subplots()


    t = start_time  # Initialize time
    while t < stop_time:
        

        v = MX.sym('v', 2*n + 7)
        v_dot = calculate_v_dot(t, v, params, controller_params, waypoint_params, p_pathframe)
        opts = {'tf': dt}
        F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
        #opts = {'tf': dt, 'linear_solver': 'csparse', 'max_num_steps': 10000}  # Example options for stiff problems
        #F = integrator('F', 'cvodes', {'x': v, 'ode': v_dot}, opts)
        r = F(x0=v0)
        v0 = r['xf']  # Update the state vector for the next iteration
        t += dt  # Increment time

        # Extract states and visualize or store data as needed...
        theta_x, theta_z, p_CM, theta_x_dot, theta_z_dot, p_CM_dot, y_int, z_int = extract_states(v0, n, '2D')

        waypoint_params, p_pathframe, target_reached = calculate_pathframe_state(p_CM, waypoint_params, controller_params, target)


        x, y, z = draw_snake_robot(ax, t, theta_x, theta_z, p_CM, params, waypoint_params, obstacles, '2D', None, alpha_h=None)
        x_list.append(x[0])
        y_list.append(y[0])
        #y_displacements_from_COM = y - p_CM[1]
        max_dist = np.max(y) - np.min(y)
        max_link_extension = np.abs(np.sin(controller_params['alpha_h']*180/np.pi)*0.14)
        print(max_link_extension - max_dist / 2)
        # Step 2: Find the maximum absolute displacement
        #max_horizontal_displacement = np.max(np.abs(y_displacements_from_COM))


        # Step 3: Print the maximum displacement
        #print(f"Maximum horizontal displacement from the COM (in y-direction): {max_horizontal_displacement} units")
        plt.pause(0.05)
      

    plt.ioff()
    # After the simulation loop
    # Plotting positions of the snake robot's links over time
    fig, ax = plt.subplots()

    # Assuming x_list and y_list are structured correctly
    for t in range(len(x_list)):
        x_positions = x_list[t]
        y_positions = y_list[t]
        ax.plot(x_positions, y_positions, 'o-', label=f'Time {t*dt:.2f}')  # dt and t used to label time

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Positions of Snake Robot Links Over Time')
    # plt.legend()  # Consider commenting this out if there are too many entries
    plt.show()

    

# Call the start_simulation function to begin the simulation
if __name__ == "__main__":
    start_simulation()



