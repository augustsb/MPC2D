
import numpy as np

def init_waypoint_parameters(waypoints):
    print("Initializing waypoint parameters...")

    waypoint_params = {}

    # Waypoints and related parameters, using NumPy for numeric computations
    #waypoints = np.array([[0, 0, 0], [2, 1, 1], [20, 1, 2], [30, 20, 3], [20, 40, 4]])
    #waypoints = np.array([[0, 0, 0], [2, 1, 1]])
  
    """
    waypoints = np.array([[0.98585795, 0.16758315,0], [2.14138391e-34, 1.67419409e-34,0]
                        ,[9.85857946e-01, 1.67583147e-01,0]
                        ,[1.98414911e+00, 1.09147237e-01,0]
                        ,[2.97745366e+00, 2.24672425e-01,0]
                        ,[3.82392578e+00, 7.57105463e-01,0]
                         ,[4.51775321e+00, 1.47724675e+00,0]
                         ,[4.83336555e+00, 2.42613496e+00,0]
                        ,[4.91185474e+00, 3.42304988e+00,0]
                        ,[4.96766418e+00, 4.42149117e+00,0]
                         ,[5.57089833e+00, 4.71511850e+00,0]
                         ,[5.00000000e+00, 5.00000000e+00,0]])
    """
  
   
    cur_WP_idx = 1  # Start from the first waypoint (considering 0-based indexing)
    r_acceptance = 0.2

    # Calculate angles between waypoints directly using NumPy
    x_prev, y_prev, z_prev = waypoints[cur_WP_idx-1]
    x_cur, y_cur, z_cur = waypoints[cur_WP_idx]

    dy = y_cur - y_prev
    dx = x_cur - x_prev
    dz = z_cur - z_prev

    cur_alpha_path = np.arctan2(dy, dx)
    cur_gamma_path = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))

    # Calculate sine and cosine directly
    sin_alpha_path = np.sin(cur_alpha_path)
    cos_alpha_path = np.cos(cur_alpha_path)
    sin_gamma_path = np.sin(cur_gamma_path)
    cos_gamma_path = np.cos(cur_gamma_path)

    # Compute rotation matrices numerically
    Rpz = np.array([[cos_alpha_path, -sin_alpha_path, 0],
                    [sin_alpha_path, cos_alpha_path, 0],
                    [0, 0, 1]])

    Rpy = np.array([[cos_gamma_path, 0, sin_gamma_path],
                    [0, 1, 0],
                    [-sin_gamma_path, 0, cos_gamma_path]])

    R_globalframe_pathframe_tr = np.dot(Rpz, Rpy).T

    # Compute the position vector of the current waypoint in the path frame
    p_WP_global = waypoints[cur_WP_idx] - waypoints[cur_WP_idx-1]
    p_WP_pathframe = np.dot(R_globalframe_pathframe_tr, p_WP_global)

    # Store numeric parameters
    waypoint_params['waypoints'] = waypoints
    waypoint_params['cur_WP_idx'] = cur_WP_idx
    waypoint_params['r_acceptance'] = r_acceptance
    waypoint_params['R_globalframe_pathframe_tr'] = R_globalframe_pathframe_tr
    waypoint_params['p_WP_global'] = p_WP_global
    waypoint_params['p_WP_pathframe'] = p_WP_pathframe
    waypoint_params['cur_alpha_path'] = cur_alpha_path
    waypoint_params['cur_gamma_path'] = cur_gamma_path

    return waypoint_params