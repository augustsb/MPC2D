
import numpy as np



def linear_push_to_optimal(current_params, optimal_params, step_size):
    """
    Gradually adjusts the current gait parameters towards the optimal values using linear steps.

    :param current_params: Dictionary of the current gait parameters ('omega_h', 'delta_h', 'alpha_h').
    :param optimal_params: Dictionary of the optimal gait parameters obtained from the optimization.
    :param step_size: Fixed step size for adjustment. Positive for increasing, negative for decreasing.
                      This can be a single value or a dictionary with specific values for each parameter.
    :return: Updated gait parameters.
    """
    updated_params = {}
    for param in ['omega_h', 'delta_h', 'alpha_h']:
        current_value = current_params[param]
        optimal_value = optimal_params[param]
        
        # Determine the direction of the adjustment
        direction = 1 if optimal_value > current_value else -1
        
        # Calculate the new value with a fixed step size
        # If step_size is a dictionary, use specific step size for each parameter
        step = step_size[param] if isinstance(step_size, dict) else step_size
        new_value = current_value + direction * min(abs(step), abs(optimal_value - current_value))
        
        updated_params[param] = new_value

    return updated_params



def exponential_push_to_optimal(current_params, optimal_params, smoothing_factor):
    """
    Gradually adjusts the current gait parameters towards the optimal values using exponential smoothing.

    :param current_params: Dictionary of the current gait parameters ('omega_h', 'delta_h', 'alpha_h').
    :param optimal_params: Dictionary of the optimal gait parameters obtained from the optimization.
    :param smoothing_factor: Smoothing factor (lambda) controlling the adjustment rate (0 < lambda < 1).
    :return: Updated gait parameters.
    """
    updated_params = {}
    for param in ['omega_h', 'delta_h', 'alpha_h']:
        current_value = current_params[param]
        optimal_value = optimal_params[param]
        new_value = current_value + smoothing_factor * (optimal_value - current_value)
        updated_params[param] = new_value
    
    return updated_params


def calculate_start_conditions(alpha_h, omega_h, delta_h, n, t):
    """
    Calculate the start conditions (position, velocity, acceleration) for each joint.

    Args:
    - alpha_h: Amplitude of the wave.
    - omega_h: Angular frequency of the wave.
    - delta_h: Phase offset between consecutive joints.
    - n: Number of joints.
    - t: Current time.

    Returns:
    - start_conditions: A list of dictionaries with 'pos', 'vel', and 'acc' for each joint.
    """
    start_conditions = []

    for i in range(n):
        # Position (angle) of the joint
        pos = alpha_h * np.sin(omega_h * t + i * delta_h)

        # Velocity of the joint
        vel = alpha_h * omega_h * np.cos(omega_h * t + i * delta_h)

        # Acceleration of the joint
        acc = -alpha_h * omega_h**2 * np.sin(omega_h * t + i * delta_h)

        start_conditions.append({'pos': pos, 'vel': vel, 'acc': acc})

    return start_conditions


def calculate_end_conditions(alpha_h_target, omega_h_target, delta_h_target, n, t, T):
    """
    Calculate the end conditions (position, velocity, acceleration) for each joint after time T.

    Args:
    - alpha_h_target: Target amplitude of the wave.
    - omega_h_target: Target angular frequency of the wave.
    - delta_h_target: Target phase offset between consecutive joints.
    - n: Number of joints.
    - t: Current time.
    - T: Duration after which end conditions are calculated.

    Returns:
    - end_conditions: A list of dictionaries with 'pos', 'vel', and 'acc' for each joint.
    """
    end_conditions = []

    for i in range(n):
        # Position (angle) at the end
        pos = alpha_h_target * np.sin(omega_h_target * (t + T) + i * delta_h_target)

        # Velocity at the end
        vel = alpha_h_target * omega_h_target * np.cos(omega_h_target * (t + T) + i * delta_h_target)

        # Acceleration at the end
        acc = -alpha_h_target * omega_h_target**2 * np.sin(omega_h_target * (t + T) + i * delta_h_target)

        end_conditions.append({'pos': pos, 'vel': vel, 'acc': acc})

    return end_conditions



def calculate_coeffs_list(start_conditions, end_conditions, T):
    """
    Calculates quintic polynomial coefficients for each joint based on start and end conditions.

    Args:
    - start_conditions: A list of dictionaries for each joint's start conditions {'pos', 'vel', 'acc'}.
    - end_conditions: A list of dictionaries for each joint's end conditions {'pos', 'vel', 'acc'}.
    - T: The duration over which to transition from start to end conditions.
    
    Returns:
    - coeffs_list: A list of coefficients for the quintic polynomial for each joint.
    """
    coeffs_list = []
    for i in range(len(start_conditions)):
        start_pos = start_conditions[i]['pos']
        start_vel = start_conditions[i]['vel']
        start_acc = start_conditions[i]['acc']
        
        end_pos = end_conditions[i]['pos']
        end_vel = end_conditions[i]['vel']
        end_acc = end_conditions[i]['acc']
        
        # Setup the equations' matrix
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        # Setup the results vector
        b = np.array([start_pos, end_pos, start_vel, end_vel, start_acc, end_acc])
        
        # Solve for the coefficients
        coeffs = np.linalg.solve(M, b)
        coeffs_list.append(coeffs)
        
    return coeffs_list






def sine_curve(A, omega, phi, t):
    """Returns the value of a sine curve at time t."""
    return A * np.sin(omega * t + phi)

def transition_alpha(start_time, end_time, current_time):
    """Calculates the blend factor alpha for the current time."""
    if current_time <= start_time:
        return 0
    elif current_time >= end_time:
        return 1
    else:
        return (current_time - start_time) / (end_time - start_time)

def blended_signal_derivatives(t, start_time, end_time, A1, omega1, phi1, A2, omega2, phi2, i):
    """Returns the blended position, velocity, and acceleration of the signal at time t."""
    alpha = transition_alpha(start_time, end_time, t)
    
    # Position (current implementation)
    y1 = A1 * np.sin(omega1 * t + i * phi1)
    y2 = A2 * np.sin(omega2 * t + i* phi2)
    blended_position = (1 - alpha) * y1 + alpha * y2
    
    # Velocity
    v1 = A1 * omega1 * np.cos(omega1 * t + i* phi1)
    v2 = A2 * omega2 * np.cos(omega2 * t + i* phi2)
    blended_velocity = (1 - alpha) * v1 + alpha * v2
    
    # Acceleration
    a1 = -A1 * omega1**2 * np.sin(omega1 * t + i* phi1)
    a2 = -A2 * omega2**2 * np.sin(omega2 * t + i* phi2)
    blended_acceleration = (1 - alpha) * a1 + alpha * a2
    
    return blended_position, blended_velocity, blended_acceleration




