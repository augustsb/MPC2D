

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import casadi as ca
from calculate_XYZ import calculate_XYZ



def draw_circle_in_3d(ax, center, radius, color='r'):
        theta = np.linspace(0, 2*np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        ax.plot(x, y,  color=color, linewidth=2)


def draw_circle_in_3d_alpha(ax, center, alpha, color='b'):
    # Assuming alpha influences radius or is the radius
    radius = alpha  # Or some function of alpha if alpha is not directly the radius
    theta = np.linspace(0, 2*np.pi, 100)  
    # Calculate circle points in 3D space
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])  # Assuming all points are at the same z level as the center
    
    # Plotting the circle in 3D
    ax.plot(x, y, z, color=color)



def draw_snake_robot(ax, t, theta_x, theta_z, p,  params,  waypoint_params, obstacles, initial_waypoints, alpha_h):

    l = params['l']
    n = params['n']
    waypoints = waypoint_params['waypoints']  # Assuming waypoints are stored in params



    # Initialize the figure
    ax.clear()

   # Set properties
    ax.set_facecolor('white')
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f't = {t:.2f}')

    
    # Calculate positions of links
    XYZ = calculate_XYZ(p, params, theta_x, theta_z)  # Assuming this function is defined elsewhere
    x, y = XYZ[:n], XYZ[n:2*n]
 
    # Draw the snake robot
    for j in range(n):

        startx = x[j] - l*np.cos(theta_z[j])*np.cos(theta_x[j])
        starty = y[j] - l*np.cos(theta_z[j])*np.sin(theta_x[j])

        endx = x[j] + l*np.cos(theta_z[j])*np.cos(theta_x[j])
        endy = y[j] + l*np.cos(theta_z[j])*np.sin(theta_x[j])


        startx = np.squeeze(startx)
        starty = np.squeeze(starty)

        endx = np.squeeze(endx)
        endy = np.squeeze(endy)

        # Plotting the links
        ax.plot([startx, endx], [starty, endy], 'bo-', linewidth=1.3)


    for obstacle in obstacles:
        draw_circle_in_3d(ax, obstacle['center'], obstacle['radius'], color='r')

    #alpha_h_flat = alpha_h.flatten()  # This converts alpha_h to a 1D array if it's not already
    # Draw amplitude circles around waypoints
    #for waypoint, alpha in zip(waypoints, alpha_h_flat):
        #draw_circle_in_3d_alpha(ax, waypoint, alpha, color='b')  # Use blue color for amplitude circles
        
    
    # Assuming waypoints is already a NumPy array as per your numeric computation adjustments
    if waypoints is not None:
        # Directly use the waypoints array for visualization
        waypoints_np = waypoints  # waypoints is already a NumPy array, no need for conversion
        # Draw waypoints if they exist
        if len(waypoints_np) > 0:  # Check if waypoints_np is not empty
            ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], color='g', marker='o', s=100, label='Waypoints')

            # Optionally, draw lines connecting waypoints to visualize the path
            if len(waypoints_np) > 1:  # Ensure there are multiple waypoints for a path
                ax.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'g--', label='Path')

    """
     # Assuming waypoints is already a NumPy array as per your numeric computation adjustments
    if initial_waypoints is not None:
        # Directly use the waypoints array for visualization
        initial_waypoints_np = initial_waypoints  # waypoints is already a NumPy array, no need for conversion
        # Draw waypoints if they exist
        if len(initial_waypoints_np) > 0:  # Check if waypoints_np is not empty
            ax.scatter(initial_waypoints_np[:, 0], initial_waypoints_np[:, 1], color='k', marker='o', s=100, label='Waypoints')

            # Optionally, draw lines connecting waypoints to visualize the path
            if len(initial_waypoints_np) > 1:  # Ensure there are multiple waypoints for a path
                ax.plot(initial_waypoints_np[:, 0], initial_waypoints_np[:, 1], 'k--', label='Path')

    """


    # Set axis limits
    p_numpy = p.full().flatten()  # Assuming p is a CasADi vector
    zoom_factor = 10  # Adjust this factor to zoom out more or less
    ax.set_xlim([p_numpy[0]-n*l-0.03, p_numpy[0]+zoom_factor*n*l+0.3])
    ax.set_ylim([p_numpy[1]-n*l-0.1, p_numpy[1]+zoom_factor*n*l+0.01])
    #ax.set_zlim([p_numpy[2]-n*l-0.01, p_numpy[2]+n*l+0.01])
    
    plt.show()






def draw_snake_robot_pcm(ax, t,  p,  params, theta_x, theta_z , waypoint_params, obstacles):

    l = params['l']
    n = params['n']
    waypoints = waypoint_params['waypoints']  # Assuming waypoints are stored in params



    # Initialize the figure
    ax.clear()

   # Set properties
    ax.set_facecolor('white')
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f't = {t:.2f}')

        # Calculate positions of links
    XYZ = calculate_XYZ(p, params, theta_x, theta_z)  # Assuming this function is defined elsewhere
    x, y = XYZ[:n], XYZ[n:2*n]



        # Draw the snake robot
    for j in range(n):
        startx = x[j] - l*np.cos(theta_x[j])
        starty = y[j] - l*np.sin(theta_x[j])

        endx = x[j] + l*np.cos(theta_x[j])
        endy = y[j] + l*np.sin(theta_x[j])


        startx = np.squeeze(startx)
        starty = np.squeeze(starty)

        endx = np.squeeze(endx)
        endy = np.squeeze(endy)

        # Plotting the links
        ax.plot([startx, endx], [starty, endy], 'bo-', linewidth=1.3)

    

    # Plotting the p_CM
    ax.plot(p[0], p[1], 'ro-', linewidth=1.3)


    for obstacle in obstacles:
        draw_circle_in_3d(ax, obstacle['center'], obstacle['radius'], color='r')
        
    
    # Assuming waypoints is already a NumPy array as per your numeric computation adjustments
    if waypoints is not None:
        # Directly use the waypoints array for visualization
        waypoints_np = waypoints  # waypoints is already a NumPy array, no need for conversion
        # Draw waypoints if they exist
        if len(waypoints_np) > 0:  # Check if waypoints_np is not empty
            ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], color='g', marker='o', s=100, label='Waypoints')

            # Optionally, draw lines connecting waypoints to visualize the path
            if len(waypoints_np) > 1:  # Ensure there are multiple waypoints for a path
                ax.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'g--', label='Path')
    
    # Set axis limits
    p_numpy = p.full().flatten()  # Assuming p is a CasADi vector
    zoom_factor = 10  # Adjust this factor to zoom out more or less
    ax.set_xlim([p_numpy[0]-n*l-0.03, p_numpy[0]+zoom_factor*n*l+0.3])
    ax.set_ylim([p_numpy[1]-n*l-0.1, p_numpy[1]+zoom_factor*n*l+0.01])
    #ax.set_zlim([p_numpy[2]-n*l-0.01, p_numpy[2]+n*l+0.01])
    
    plt.show()





