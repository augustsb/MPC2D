
from sklearn.metrics import mean_squared_error
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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
X = data[['alpha_h', 'omega_h','delta_h', 'average_velocity']]
y = data[['average_energy']] 



# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X
y_train = y


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