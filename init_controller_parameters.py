import numpy as np

def init_controller_parameters(n,l):
    print("Initializing controller parameters...")

    # Initialize controller parameters dictionary
    controller_params = {}

    # Joint controller parameters
    #controller_params['Kp_joint'] = 200
    #controller_params['Kd_joint'] = 50
    controller_params['Kp_joint'] = 200
    controller_params['Kd_joint'] = 50

    # Parameters for gait pattern horizontal
    #controller_params['alpha_h'] = 30 * np.pi / 180
    #controller_params['omega_h'] = 150 * np.pi / 180
    #controller_params['delta_h'] = 40 * np.pi / 180

    controller_params['transition_in_progress'] = False

    controller_params['alpha_h'] = 0.7155849933176757
    controller_params['omega_h'] = 0.5235987755982988
    controller_params['delta_h'] = 0.6981317007977318



    # Parameters for gait pattern vertical
    controller_params['alpha_v'] = 10 * np.pi / 180
    controller_params['omega_v'] = 150 * np.pi / 180
    controller_params['delta_v'] = 40 * np.pi / 180


    #ILOS parameters
    controller_params['K_theta'] = 0.25
    controller_params['K_psi'] = 0.2
    controller_params['sigma_y'] = 0.12
    controller_params['sigma_z'] = 0.09
    controller_params['delta_z'] = n*l
    #controller_params['delta_y'] = 4*n*l
    controller_params['delta_y'] =  n*l
    controller_params['z_int0'] = 0
    controller_params['y_int0'] = 0

    


    return controller_params