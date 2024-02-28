import casadi as ca

def calculate_cos_theta_diag(theta):
    """
    Returns a diagonal matrix of the cosine values of the joint angles.

    :param theta: CasADi MX or SX matrix of joint angles
    :return: CasADi MX or SX diagonal matrix of cosine values
    """
    # Use CasADi's cos function directly on theta
    cos_theta = ca.cos(theta)
    # Create a diagonal matrix
    cos_theta_diag = ca.diag(cos_theta)
    return cos_theta_diag

def calculate_sin_theta_diag(theta):
    """
    Returns a diagonal matrix of the sine values of the joint angles.

    :param theta: CasADi MX or SX matrix of joint angles
    :return: CasADi MX or SX diagonal matrix of sine values
    """
    # Use CasADi's sin function directly on theta
    sin_theta = ca.sin(theta)
    # Create a diagonal matrix
    sin_theta_diag = ca.diag(sin_theta)
    return sin_theta_diag