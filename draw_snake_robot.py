

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


def draw_sphere(ax, center, radius, color='r'):
    # Generate points for a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=1.0, zorder=2)



def draw_snake_robot(ax, target, t, theta_x, theta_z, p,  params,  waypoint_params, obstacles, dimension,  initial_waypoints, alpha_h):

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

    x, y, z = XYZ[:n], XYZ[n:2*n],  XYZ[2*n:3*n]

    #middle_link_index = n // 2  # This is an integer division
 
    # Draw the snake robot
    for j in range(n):

        startx = x[j] - l*np.cos(theta_z[j])*np.cos(theta_x[j])
        starty = y[j] - l*np.cos(theta_z[j])*np.sin(theta_x[j])
        startz = z[j] + l*np.sin(theta_z[j])
        endx = x[j] + l*np.cos(theta_z[j])*np.cos(theta_x[j])
        endy = y[j] + l*np.cos(theta_z[j])*np.sin(theta_x[j])
        endz = z[j] - l*np.sin(theta_z[j])


        startx = np.squeeze(startx)
        starty = np.squeeze(starty)
        startz = np.squeeze(startz)
        endx = np.squeeze(endx)
        endy = np.squeeze(endy)
        endz = np.squeeze(endz)


        if dimension == '3D':
        # Plotting the links
            ax.plot([startx, endx], [starty, endy], [startz, endz], 'bo-', linewidth=1.0, zorder=10)
            ax.plot([target[0]], [target[1]], [target[2]], 'bo', markersize=2, label='Goal')

        if dimension == '2D':
            ax.plot([startx, endx], [starty, endy], 'bo-', linewidth=1.3)
            ax.plot([target[0]], [target[1]], 'bo', markersize=2, label='Goal')


    
    for obstacle in obstacles:
        if dimension == '2D':
            draw_circle_in_3d(ax, obstacle['center'], obstacle['radius'], color='r')

        if dimension == '3D':
            draw_sphere(ax, obstacle['center'], obstacle['radius'], color='r')


    # Assuming waypoints is already a NumPy array as per your numeric computation adjustments
    if waypoints is not None:
        # Directly use the waypoints array for visualization
        waypoints_np = waypoints  # waypoints is already a NumPy array, no need for conversion
        # Draw waypoints if they exist
        if len(waypoints_np) > 0:  # Check if waypoints_np is not empty

            if dimension == '3D':
                ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], waypoints_np[:, 2], color='g', marker='o', s=100, label='Waypoints')
            
            if dimension == '2D':
                ax.scatter(waypoints_np[:, 0], waypoints_np[:, 1], color='g', marker='o', s=100, label='Waypoints')

            # Optionally, draw lines connecting waypoints to visualize the path
            if len(waypoints_np) > 1:  # Ensure there are multiple waypoints for a path

                if dimension == '3D':
                    ax.plot(waypoints_np[:, 0], waypoints_np[:, 1], waypoints_np[:, 2], 'g--', label='Path')
                
                if dimension == '2D':
                    ax.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'g--', label='Path')


    # Set axis limits
    p_numpy = p.full().flatten()  # Assuming p is a CasADi vector
    zoom_factor = 20  # Adjust this factor to zoom out more or less
    ax.set_xlim([p_numpy[0]-n*l-0.03, p_numpy[0]+zoom_factor*n*l+0.3])
    ax.set_ylim([p_numpy[1]-n*l-0.1, p_numpy[1]+zoom_factor*n*l+0.01])
    if dimension == '3D':
        ax.set_zlim([p_numpy[2]-n*l-0.1, p_numpy[2]+zoom_factor*n*l+0.01])

    plt.show()
    #middle_link_position = np.array((x[middle_link_index], y[middle_link_index], z[middle_link_index])).flatten()
    return x, y, z







