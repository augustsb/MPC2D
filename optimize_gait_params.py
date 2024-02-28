

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor




# Sample data for lateral undulation and eel-like motion
# wp α [deg] ω [deg/s] δ [deg] υ¯ [m/s] Pavg [W]
data_lateral_undulation = np.array([
    [0,     44.0100, 210.0000, 15.1400, 0.8425, 34.2515],
    [0.0500, 41.5451, 209.9863, 16.1863, 0.8407, 29.9318],
    [0.1000, 38.5042, 209.9987, 17.6406, 0.8327, 24.9031],
    [0.1500, 37.1659, 209.9976, 18.9259, 0.8259, 22.3614],
    [0.2000, 34.9161, 209.6835, 20.3226, 0.8107, 18.9214],
    [0.2500, 33.4248, 209.9899, 22.9651, 0.7922, 15.6914],
    [0.3000, 32.1065, 209.9997, 26.1707, 0.7640, 12.4822],
    [0.3500, 31.7042, 209.9932, 28.3385, 0.7444, 10.8197],
    [0.4000, 31.9861, 209.9938, 30.8753, 0.7233, 9.3646],
    [0.4500, 32.0642, 209.9931, 33.6360, 0.6956, 7.8220],
    [0.5000, 32.4826, 209.9995, 35.8916, 0.6731, 6.7981],
    [0.5500, 33.2925, 209.9890, 38.9056, 0.6417, 5.6396],
    [0.6000, 34.4218, 209.9967, 41.3291, 0.6170, 4.8916],
    [0.6500, 35.0257, 209.9667, 44.0756, 0.5842, 4.0883],
    [0.7000, 37.7160, 209.9899, 47.2997, 0.5523, 3.4617],
    [0.7500, 39.7087, 207.8852, 50.9907, 0.5060, 2.7433],
    [0.8000, 39.1360, 193.1456, 53.9248, 0.4379, 1.9467],
    [0.8500, 54.4381, 209.6399, 75.3818, 0.3102, 0.8571],
    [0.9000, 68.4280, 207.9591, 89.1905, 0.2425, 0.4111],
    [0.9500, 46.4469, 138.4423, 89.9360, 0.1555, 0.1693],
    [1.0000, 0, 16.5134, 59.8411, 0, 0]
])

# wp α [deg] ω [deg/s] δ [deg] υ¯ [m/s] Pavg [W]
# Eel-like motion data
data_eel = np.array([
    [0, 59.2350, 209.9813, 26.1994, 0.6038, 13.4369],
    [0.0500, 58.9029, 209.9887, 26.5538, 0.6037, 13.1134],
    [0.1000, 56.7018, 209.9712, 28.9106, 0.6004, 11.0886],
    [0.1500, 54.6536, 209.9960, 30.3223, 0.5960, 9.7995],
    [0.2000, 53.8984, 209.9943, 31.4665, 0.5922, 9.0650],
    [0.2500, 52.6137, 209.9798, 33.4673, 0.5838, 7.8826],
    [0.3000, 52.1308, 209.9987, 34.8301, 0.5776, 7.2274],
    [0.3500, 51.3303, 209.9852, 36.6885, 0.5675, 6.3739],
    [0.4000, 51.0358, 209.9947, 38.4587, 0.5576, 5.7077],
    [0.4500, 51.4150, 209.9983, 40.4453, 0.5469, 5.1124],
    [0.5000, 52.2181, 209.9899, 42.9157, 0.5327, 4.4629],
    [0.5500, 51.7512, 209.9984, 44.8704, 0.5178, 3.9089],
    [0.6000, 53.5402, 209.9995, 47.6477, 0.5007, 3.3852],
    [0.6500, 56.0090, 209.9980, 51.2005, 0.4769, 2.7965],
    [0.7000, 57.0378, 209.9775, 53.9967, 0.4552, 2.3694],
    [0.7500, 62.0106, 209.9535, 58.4522, 0.4245, 1.8881],
    [0.8000, 65.6760, 209.9181, 62.8889, 0.3915, 1.4920],
    [0.8500, 73.7492, 209.8543, 71.2193, 0.3337, 0.9857],
    [0.9000, 89.2687, 209.9100, 89.9740, 0.2334, 0.3958],
    [0.9500, 75.0664, 150.9382, 89.7837, 0.1663, 0.2004],
    [1.0000, 1.0695, 0.0639, 51.8275, 0, 0]
])



"""

# Assuming data_lateral_undulation and data_eel are already defined as shown previously

# Split the data for lateral undulation
X_lateral = data_lateral_undulation[:, :4]  # Features: wp, α, ω, δ
y_speed_lateral = data_lateral_undulation[:, 4]  # Target: ῡ (average forward velocity)
y_power_lateral = data_lateral_undulation[:, 5]  # Target: Pavg (power consumption)

# Split the data for eel-like motion
X_eel = data_eel[:, :4]  # Features: wp, α, ω, δ
y_speed_eel = data_eel[:, 4]  # Target: ῡ (average forward velocity)
y_power_eel = data_eel[:, 5]  # Target: Pavg (power consumption)

# Train models for lateral undulation
model_speed_lateral = LinearRegression().fit(X_lateral, y_speed_lateral)
model_power_lateral = LinearRegression().fit(X_lateral, y_power_lateral)

# Train models for eel-like motion
model_speed_eel = LinearRegression().fit(X_eel, y_speed_eel)
model_power_eel = LinearRegression().fit(X_eel, y_power_eel)

# Save the models
dump(model_speed_lateral, 'model_speed_lateral.joblib')
dump(model_power_lateral, 'model_power_lateral.joblib')
dump(model_speed_eel, 'model_speed_eel.joblib')
dump(model_power_eel, 'model_power_eel.joblib')

# Assuming model is your trained LinearRegression model
beta_0_speed = model_speed_lateral.intercept_
betas_speed = model_speed_lateral.coef_

beta_0_power = model_power_lateral.intercept_
betas_power = model_power_lateral.coef_

print(beta_0_speed, betas_speed)
print(beta_0_power, betas_power)


print(beta_0_speed + np.dot(np.array(betas_speed), np.array([0.5, 32, 209, 36])))


"""



# Split the data for lateral undulation
X_lateral = data_lateral_undulation[:, 1:4]  # Features: wp, α, ω, δ
y_speed_lateral = data_lateral_undulation[:, 4]  # Target: ῡ (average forward velocity)
y_power_lateral = data_lateral_undulation[:, 5]  # Target: Pavg (power consumption)

# Split the data for eel-like motion
X_eel = data_eel[:, 1:4]  # Features: wp, α, ω, δ
y_speed_eel = data_eel[:, 4]  # Target: ῡ (average forward velocity)
y_power_eel = data_eel[:, 5]  # Target: Pavg (power consumption)

# Train models for lateral undulation using Random Forest
model_speed_lateral_rf = RandomForestRegressor(n_estimators=100).fit(X_lateral, y_speed_lateral)
model_power_lateral_rf = RandomForestRegressor(n_estimators=100).fit(X_lateral, y_power_lateral)

# Train models for eel-like motion using Random Forest
model_speed_eel_rf = RandomForestRegressor(n_estimators=100).fit(X_eel, y_speed_eel)
model_power_eel_rf = RandomForestRegressor(n_estimators=100).fit(X_eel, y_power_eel)


X = data_lateral_undulation[:,1:4]  # gait parameters: α, ω, δ
y = data_lateral_undulation[:,5]    # energy consumption
y_speed = data_lateral_undulation[:,4]

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit a linear model on the polynomial features
model = LinearRegression().fit(X_poly, y)

# Extract the model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print(coefficients)
print(intercept)


# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit a linear model on the polynomial features
model_speed = LinearRegression().fit(X_poly, y_speed)

# Extract the model coefficients and intercept
coefficients_speed = model_speed.coef_
intercept_speed = model_speed.intercept_

print(coefficients_speed)
print(intercept_speed)