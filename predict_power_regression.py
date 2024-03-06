

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline



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






# Assuming all files are in your specified directory
directory_path = "C:\\Users\\augus\\OneDrive\\Documents\\Master2024code\\MPC_2D\\results_2802"
file_prefix = "chunk_results_"
all_files = [os.path.join(directory_path, f"{file_prefix}{i}.csv") for i in range(16)]

# Read and concatenate all files into a single DataFrame
df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)

# Adjusting 'average_energy' and 'average_power' based on 'success'
data.loc[~data['success'], ['average_energy', 'average_velocity']] = 0

# Proceed with selecting the features and target variable
X = data[['alpha_h', 'omega_h', 'delta_h']]
y = data[['average_energy', 'average_velocity']]  # Or 'average_energy', depending on which one you're predicting


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initializing the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)


# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error and R-squared values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")




# Initialize the model: Random forest

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model_rf.fit(X_train, y_train)

# Make predictions
y_pred = model_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

from joblib import dump

# Assuming model_rf is your trained Random Forest model
dump(model_rf, 'model_rf.joblib')


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Create a polynomial regression model
degree = 2  # You can adjust the degree based on model performance
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the model
poly_model.fit(X, y)

y_pred_poly = poly_model.predict(X_test)

# Calculate the Mean Squared Error
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Calculate the R-squared value
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression Model - Mean Squared Error: {mse_poly}")
print(f"Polynomial Regression Model - R-squared: {r2_poly}")

# Extract the PolynomialFeatures transformer and LinearRegression model from the pipeline
poly_features = poly_model.named_steps['polynomialfeatures']
linear_model = poly_model.named_steps['linearregression']

# Get the coefficients (weights) and intercept
coefficients = linear_model.coef_
intercept = linear_model.intercept_

print(coefficients)
print(intercept)




