from change_gait_params import calculate_start_conditions, calculate_end_conditions, calculate_coeffs_list
import numpy as np
import matplotlib.pyplot as plt


alpha_h_current = 30 * np.pi / 180
omega_h_current = 150 * np.pi / 180
delta_h_current = 40 * np.pi / 180
n = 10

alpha_h_target = 10 * np.pi / 180
omega_h_target = 210 * np.pi / 180
delta_h_target = 40 * np.pi / 180

t = 0
T = 1

start_conditions = calculate_start_conditions(alpha_h_current, omega_h_current, delta_h_current, n, t)
end_conditions = calculate_end_conditions(alpha_h_target, omega_h_target, delta_h_target, n, t, T)
coeffs_list = calculate_coeffs_list(start_conditions, end_conditions, T)

time_steps = []
phi_refs = []
# Increment t in the loop
dt = 0.1  # Time step increment
while t < 20:
    elapsed_time = t
    if t > T:
        for i in range(n):
            alpha_h_current = alpha_h_target
            omega_h_current = omega_h_target
            delta_h_current = delta_h_target
            phi_ref = alpha_h_current * np.sin(omega_h_current * t + i * delta_h_current)
            # Assuming you want to plot the last phi_ref of each time step
    else:
        phi_ref = 0
        for i in range(n):
            coeffs = coeffs_list[i]
            current_pos = coeffs[0] + coeffs[1] * elapsed_time + coeffs[2] * elapsed_time**2 + coeffs[3] * elapsed_time**3 + coeffs[4] * elapsed_time**4 + coeffs[5] * elapsed_time**5
            phi_ref = current_pos  # Assuming you want to plot the last current_pos of each time step
    
    # Append current time and phi_ref to their respective lists
    time_steps.append(t)
    phi_refs.append(phi_ref)
    
    t += dt  # Increment time


# Convert phi_refs to a numpy array for element-wise operations
phi_refs_array = np.array(phi_refs)
# Now you can apply the conversion to degrees directly on the array
phi_refs_degrees = phi_refs_array * 180 / np.pi

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_steps, phi_refs_degrees, label="phi_ref over time (Degrees)")
plt.xlabel("Time")
plt.ylabel("phi_ref (Degrees)")
plt.title("Transition of Gait Parameters")
plt.legend()
plt.grid(True)
plt.show()

 








