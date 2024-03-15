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