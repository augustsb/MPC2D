
import numpy as np


def calculate_pathframe_state(p_CM, waypoint_params, controller_params, target):


    
    waypoints = np.array(waypoint_params['waypoints'])
    cur_WP_idx = waypoint_params['cur_WP_idx']
    r_acceptance = waypoint_params['r_acceptance']

    
    # Compute the vector from the path frame to the CM of the snake in the global frame
    p_pathframe_global = p_CM - waypoints[cur_WP_idx-1]
    
    # Assuming R_globalframe_pathframe_tr is a rotation matrix calculated elsewhere and passed here
    R_globalframe_pathframe_tr = waypoint_params['R_globalframe_pathframe_tr']
    p_pathframe = np.dot(R_globalframe_pathframe_tr, p_pathframe_global)
    p_pathframe = p_pathframe.flatten()
    
    p_WP_pathframe = waypoint_params['p_WP_pathframe']  # Assuming this is already calculated and stored
    
    # Compute distance to waypoint
    WP_dist = np.abs(p_WP_pathframe - p_pathframe)
    distance = np.sqrt(np.sum(WP_dist**2))
    
    # Determine if the current waypoint has been reached
    activate_next_WP = distance <= r_acceptance

    vec_to_target = np.abs(p_CM - target)
    dist_to_target = np.sqrt(np.sum(vec_to_target**2))
    target_reached = dist_to_target <= r_acceptance

    
    if activate_next_WP and cur_WP_idx < len(waypoints) - 1:
        cur_WP_idx += 1
        
        # Calculate the angle of the straight path between the previous and current waypoint
        x_prev, y_prev, z_prev = waypoints[cur_WP_idx-1]
        x_cur, y_cur, z_cur = waypoints[cur_WP_idx]
        
        dy = y_cur - y_prev
        dx = x_cur - x_prev
        dz = z_cur - z_prev
        
        #cur_alpha_path = np.arctan2(dy, dx)
        #cur_gamma_path = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
        cur_alpha_path = detect_yaw_pos_discontinuity(np.arctan2(dy, dx), waypoint_params.get('cur_alpha_path', 0))
        cur_gamma_path = detect_yaw_pos_discontinuity(np.arctan2(-dz, np.sqrt(dx**2 + dy**2)), waypoint_params.get('cur_gamma_path', 0))

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
        waypoint_params['cur_alpha_path'] = cur_alpha_path
        waypoint_params['cur_gamma_path'] = cur_gamma_path
        waypoint_params['p_WP_pathframe'] = p_WP_pathframe
        waypoint_params['R_globalframe_pathframe_tr'] = R_globalframe_pathframe_tr
        waypoint_params['cur_WP_idx'] = cur_WP_idx
 
        # Update angles in waypoint_params


    return waypoint_params, p_pathframe, target_reached



def detect_yaw_pos_discontinuity(_in, prev):
    two_pi = 2 * np.pi
    mult = np.floor(prev / two_pi)
    mult2Pi = mult * two_pi
    output = _in + mult2Pi

    if prev - output > np.pi:
        output += two_pi
    if output - prev > np.pi:
        output -= two_pi

    return output



