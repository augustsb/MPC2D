

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
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train_models import train_linear_regression, train_random_forest, train_polynomial_regression, train_neural_network
from pymoo.problems.multi import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# Sample data for lateral undulation (ELENI)
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
# Eel-like motion data (ELENI)
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

# α [deg] ω [deg/s] δ [deg] υ¯ [m/s] Pavg [W] MY OWN PARETO FRONT
data = np.array([
    [0.087266, 3.665191, 0.261799, 0.445039, 0.284728],
    [0.174533, 2.356194, 0.523599, 0.449165, 0.369763],
    [0.174533, 2.443461, 0.523599, 0.464443, 0.391545],
    [0.174533, 2.530727, 0.523599, 0.478966, 0.417391],
    [0.174533, 2.617994, 0.523599, 0.493197, 0.445189],
    [0.174533, 2.705260, 0.523599, 0.507383, 0.482964],
    [0.087266, 3.665191, 0.174533, 0.400988, 0.520056],
    [0.174533, 2.792527, 0.523599, 0.521817, 0.527017],
    [0.174533, 2.879793, 0.523599, 0.536896, 0.576738],
    [0.174533, 2.967060, 0.523599, 0.552032, 0.619511],
    [0.174533, 3.054326, 0.523599, 0.567545, 0.658881],
    [0.174533, 3.141593, 0.523599, 0.582752, 0.690708],
    [0.174533, 3.228859, 0.523599, 0.597836, 0.716678],
    [0.174533, 3.316126, 0.523599, 0.611778, 0.743627],
    [0.087266, 3.665191, 0.087266, 0.283402, 0.751021],
    [0.174533, 3.403392, 0.523599, 0.625710, 0.774573],
    [0.174533, 3.490659, 0.523599, 0.639436, 0.817025],
    [0.174533, 3.577925, 0.523599, 0.653297, 0.871023],
    [0.174533, 3.665191, 0.523599, 0.667533, 0.929634],
    [0.174533, 2.530727, 0.436332, 0.804235, 1.062600],
    [0.174533, 2.617994, 0.436332, 0.809632, 1.169500],
    [0.174533, 2.705260, 0.436332, 0.820248, 1.352570],
    [0.174533, 2.792527, 0.436332, 0.825903, 1.458270],
    [0.174533, 2.879793, 0.436332, 0.838594, 1.674490],
    [0.174533, 2.967060, 0.436332, 0.848548, 1.760280],
    [0.174533, 3.577925, 0.436332, 0.920900, 1.788800],
    [0.174533, 3.665191, 0.436332, 0.931162, 1.795350],
    [0.261799, 3.403392, 0.523599, 0.936544, 4.702030],
    [0.261799, 3.490659, 0.523599, 0.945509, 4.801790],
    [0.261799, 3.577925, 0.523599, 0.955418, 4.978710],
    [0.261799, 3.665191, 0.523599, 0.965732, 5.198850]
])


"""
def load_and_preprocess_data(directory_path, file_prefix, num_files):
    all_files = [os.path.join(directory_path, f"{file_prefix}{i}.csv") for i in range(num_files)]
    df_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(df_list, ignore_index=True)
    data.loc[~data['success'], ['average_energy', 'average_velocity']] = 0
    return data

"""
"""
def find_optimal_configuration(data_all, alpha_h_constraint, V_min, V_max, sol_V, prei):
    # Ensure correct logical operations for filtering with Pandas
    alpha_h_i_value = float(alpha_h_constraint)

    valid_entries = data_all[(data_all['success'] == True) & 
                             (data_all['alpha_h'] <= alpha_h_i_value) & 
                             (abs(data_all['average_velocity']) >= V_min) & 
                             (abs(data_all['average_velocity']) <= V_max)]

    #valid_entries = data_all[(data_all['alpha_h'] <= alpha_h_i_value)]

    
    if not valid_entries.empty:
        # Find the entry with minimum average energy
        #optimal_entry = valid_entries.loc[abs(valid_entries['average_energy']).idxmin()]

        valid_entries['velocity_diff'] = np.abs(valid_entries['average_velocity'] - sol_V)
        optimal_entry = valid_entries.loc[valid_entries['velocity_diff'].idxmin()]
        #return optimal_entry.drop('velocity_diff', axis=1)
        return optimal_entry

    else:
        # Second attempt: No valid entries found; look for the closest entry above the alpha_h constraint
        closest_above_entries = data_all[(data_all['alpha_h'] > alpha_h_i_value) &
                                          (abs(data_all['average_velocity']) >= V_min) & 
                                          (abs(data_all['average_velocity']) <= V_max)]
        
        if not closest_above_entries.empty:
            # Calculate the difference without adding it to the DataFrame
            closest_above_entries['alpha_h_diff'] = closest_above_entries['alpha_h'] - alpha_h_i_value
            min_diff_index = closest_above_entries['alpha_h_diff'].idxmin()
            closest_entry = closest_above_entries.loc[min_diff_index]
            # No need to drop 'alpha_h_diff' since we're not returning the entire DataFrame
            return closest_entry
        else:
            # No entries found that satisfy the velocity constraints
            return None
        

"""

"""
def find_optimal_configuration(data_all, alpha_h_constraint, V_min, V_max, sol_V, predicted_energy):
    alpha_h_i_value = float(alpha_h_constraint)

    valid_entries = data_all[(data_all['success'] == True) & 
                             (data_all['alpha_h'] <= alpha_h_i_value) & 
                             (np.abs(data_all['average_velocity']) >= V_min) & 
                             (np.abs(data_all['average_velocity']) <= V_max)]

    if not valid_entries.empty:
        # Calculate the difference in velocity and energy from the solution
        valid_entries['velocity_diff'] = np.abs(valid_entries['average_velocity'] - sol_V)
        valid_entries['energy_diff'] = np.abs(valid_entries['average_energy'] - predicted_energy)

        # Combine the differences into a single metric, e.g., by summing them
        # You might need to normalize or scale these differences if they are on very different scales
        valid_entries['combined_diff'] = valid_entries['velocity_diff'] + valid_entries['energy_diff']
        
        # Find the entry with the minimum combined difference
        optimal_entry = valid_entries.loc[valid_entries['combined_diff'].idxmin()]
        return optimal_entry.drop(['velocity_diff', 'energy_diff', 'combined_diff'], axis=1, errors='ignore')

    else:
        # If no valid entries are found in the first attempt, you can either return None or 
        # implement logic to relax constraints slightly and retry, depending on your application's needs.
        return None

"""

def find_optimal_configuration(data_all, alpha_h_constraint, V_min, V_max):
    # Ensure 'predicted_energy' is a scalar if it's wrapped in a list or similar

    # Filter entries based on 'alpha_h', 'V_min', and 'V_max'
    valid_entries = data_all[
        (data_all['success'] == True) &
        (data_all['alpha_h'] <= alpha_h_constraint) &
        (data_all['average_velocity'] >= V_min) &
        (data_all['average_velocity'] <= V_max)
    ]

    valid_entries = valid_entries.copy()

    if not valid_entries.empty:
        optimal_entry_index = valid_entries['average_energy'].abs().idxmin()
        optimal_entry = valid_entries.loc[optimal_entry_index]
        #return optimal_entry.drop('velocity_diff', axis=1)
        return optimal_entry
    
    else:
        return None
        


def load_and_preprocess_data(directory_path, file_prefix, num_files):
    all_files = [os.path.join(directory_path, f"{file_prefix}{i}.csv") for i in range(num_files)]
    df_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(df_list, ignore_index=True)
    # Assuming success is defined as having non-zero average_energy and average_velocity
    # This line might be redundant if 'success' accurately flags all and only successful instances
    #data = data[(data['success']) == True & (data['delta_h'] == 0.523599)] 

    data = data[(data['success']) == True] 
    #data['average_energy'] = data['average_energy'].abs()
    #data['average_velocity'] = data['average_velocity'].abs()

    return data



def get_features_targets(data):
    X = data[['alpha_h', 'omega_h', 'delta_h', 'average_velocity']]
    y = data[['average_energy']]
    return X, y

def get_features_targets_alpha(data):
    X = data[['predicted_energy']]
    y = data[['average_energy']] 
    return X, y


def predict_energy_using_poly_model(alpha_h, omega_h, delta_h, V, poly_model):
    # Note: This is a simplified example. You'd need to adapt it based on your actual model input format.
    # Assume the model expects a 2D array [[alpha_h, omega_h, delta_h, V]] for a single prediction.
    input_features = np.array([[alpha_h, omega_h, delta_h, V]])
    return poly_model.predict(input_features)[0]  # Assuming the model returns a 2D array and we're interested in the first value.



def predict_energy_using_poly_model_alpha(alpha_h, V, poly_model):
    # Note: This is a simplified example. You'd need to adapt it based on your actual model input format.
    # Assume the model expects a 2D array [[alpha_h, omega_h, delta_h, V]] for a single prediction.
    input_features = np.array([[alpha_h, V]])
    return poly_model.predict(input_features)[0]  # Assuming the model returns a 2D array and we're interested in the first value.


def make_pareto_front(data):
    # Negate average_velocity for "maximization"
    objectives = data[['average_velocity', 'average_energy']].copy()
    objectives['average_velocity'] = -objectives['average_velocity']
    
    # Perform non-dominated sorting on the objectives
    nds = NonDominatedSorting().do(objectives.to_numpy(), only_non_dominated_front=True)
    
    # Select rows that are part of the Pareto front
    pareto_front_rows = data.iloc[nds]
    
    # Correctly negate average_velocity back after sorting
    #pareto_front_rows['average_velocity'] = -pareto_front_rows['average_velocity']
    
    # Create DataFrame with only the intended columns
    pareto_df = pareto_front_rows[['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'average_energy']]
    
    return pareto_df


def make_pareto_front_pred(data_pred):
    # Negate average_velocity for "maximization"
    objectives = data_pred[['average_velocity', 'predicted_energy']].copy()
    objectives['average_velocity'] = -objectives['average_velocity']
    
    # Perform non-dominated sorting on the objectives
    nds = NonDominatedSorting().do(objectives.to_numpy(), only_non_dominated_front=True)
    
    # Select rows that are part of the Pareto front
    pareto_front_rows = data_pred.iloc[nds]
    
    # Correctly negate average_velocity back after sorting
    #pareto_front_rows['average_velocity'] = -pareto_front_rows['average_velocity']
    
    # Create DataFrame with only the intended columns
    pareto_df = pareto_front_rows[['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'predicted_energy']]
    
    return pareto_df


    """
    # Since we're minimizing average_energy and maximizing average_velocity, ensure correct orientation
    plt.scatter(pareto_df['average_velocity'], pareto_df['average_energy'])
    plt.xlabel('Average Velocity')
    plt.ylabel('Average Energy')
    plt.title('Pareto Front')
    plt.gca().invert_yaxis()  # If lower average_energy values are better
    plt.show()

    # Extract objectives and parameters
    objectives_and_params = data[['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'average_energy']].to_numpy()
    # In this context, assuming we want to maximize average_velocity and minimize average_energy
    # Negate average_velocity for "maximization" since pymoo performs minimization
    objectives_and_params[:, 3] = -objectives_and_params[:, 3]
    # Perform non-dominated sorting
    nds = NonDominatedSorting().do(objectives_and_params[:, -2:], only_non_dominated_front=True)
    # Extract non-dominated points including parameters
    pareto_points_with_params = objectives_and_params[nds]
    # Convert back the negation for average_velocity for visualization or further analysis
    pareto_points_with_params[:, 3] = -pareto_points_with_params[:, 3]
    pareto_df = pd.DataFrame(pareto_points_with_params, columns=['alpha_h', 'omega_h', 'delta_h', 'average_velocity', 'average_energy'])
    pareto_df_sorted = pareto_df.sort_values(by='average_energy', ascending=True)
    pareto_df_sorted['average_energy'] = pareto_df_sorted['average_energy'].abs()
    pareto_df_sorted = pareto_df_sorted.sort_values(by='average_energy', ascending=True)
    print(pareto_df_sorted)
    """

def plot_pareto_fronts(actual_pareto_df, predicted_pareto_points):
    plt.figure(figsize=(10, 6))

    # Plot actual Pareto front
    plt.scatter(actual_pareto_df['average_velocity'], actual_pareto_df['average_energy'], color='blue', label='Actual Pareto Front')

    # Plot predicted Pareto front
    plt.scatter(predicted_pareto_points['average_velocity'], predicted_pareto_points['predicted_energy'], color='red', label='Predicted Pareto Front', marker='x')

    plt.xlabel('Average Velocity')
    plt.ylabel('Average Energy')
    plt.title('Predicted vs. Actual Pareto Front')
    plt.gca().invert_yaxis()  # Assuming lower average_energy values are better
    plt.legend()
    plt.show()


def predict_and_save(directory_path, file_prefix, num_files, poly_model, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    for i in range(num_files):
        file_path = os.path.join(directory_path, f"{file_prefix}{i}.csv")
        df = pd.read_csv(file_path)
        
        # Assuming 'alpha_h', 'omega_h', 'delta_h', 'average_velocity' are your features
        #features = df[['alpha_h', 'omega_h', 'delta_h', 'average_velocity']]
        features = df[['alpha_h', 'average_velocity']]
        
        # Predict energy for each row in the DataFrame
        predicted_energies = poly_model.predict(features)
        
        # Add the predictions to the DataFrame
        df['predicted_energy'] = predicted_energies
        
        # Define the output file path
        output_file_path = os.path.join(output_directory, f"predicted_{file_prefix}_{i}.csv")
        
        # Save the DataFrame with predictions to a new file
        df.to_csv(output_file_path, index=False)

        print(f"Processed and saved predictions for {file_path}")



def plot_gait_parameter_influence(pareto_actual, pareto_predicted, parameter_name):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting for the actual Pareto front
    color = 'tab:blue'
    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel('Velocity', color=color)
    ax1.scatter(pareto_actual[parameter_name], pareto_actual['average_velocity'], color=color, label='Actual Velocity')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Energy', color=color)  # we already handled the x-label with ax1
    ax2.scatter(pareto_actual[parameter_name], pareto_actual['average_energy'], color=color, label='Actual Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # to make sure the layout doesn't get messed up
    plt.title(f'Actual Data: {parameter_name} Influence on Velocity and Energy')
    plt.show()


def plot_predicted_energy(model, alpha_h_range, omega_h_range, delta_h_range, V_range, typical_values):
    plt.figure(figsize=(14, 7))
    parameters = [alpha_h_range, omega_h_range, delta_h_range, V_range]
    param_names = ['alpha_h', 'omega_h', 'delta_h', 'V']
    
    for i, param_range in enumerate(parameters):
        plt.subplot(2, 2, i+1)
        y_values = []
        
        for param_value in param_range:
            # Prepare the feature vector for prediction
            features = np.array([[typical_values['alpha_h'], typical_values['omega_h'], typical_values['delta_h'], typical_values['average_velocity']]])
            features[0][i] = param_value  # Update the i-th parameter with the current value from its range

            # Check if the model is a PyTorch model
            if isinstance(model, torch.nn.Module):
                # Convert features to a PyTorch tensor
                features_tensor = torch.tensor(features, dtype=torch.float32)
                # Use the model to predict, ensuring it's in no_grad context for inference
                with torch.no_grad():
                    predicted_energy = model(features_tensor).numpy().flatten()[0]
            else:
                # Use the predict method directly for sklearn models
                predicted_energy = model.predict(features).flatten()[0]
            
            y_values.append(predicted_energy)
        
        plt.plot(param_range, y_values, label=f'Energy vs. {param_names[i]}')
        plt.xlabel(param_names[i])
        plt.ylabel('Predicted Energy')
        plt.title(f'Predicted Energy vs. {param_names[i]}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    directory_path = "/home/augustsb/MPC2D/results_2802"
    directory_path_predictions = "/home/augustsb/MPC2D/prediction_results_2802"
    directory_path_reprocessed = "/home/augustsb/MPC2D/reprocessed_results_2802"
    directory_path_predictions_reprocessed = "/home/augustsb/MPC2D/predictions_reprocessed_results_2802"

    file_prefix = "chunk_results_"
    file_prefix_predictions = "predicted_chunk_results_"
    file_prefix_reprocessed = "reprocessed_chunk_results_"



    data = load_and_preprocess_data(directory_path, file_prefix, 16)
    data_pred = load_and_preprocess_data(directory_path_predictions, file_prefix_predictions, 16)
    data_new = load_and_preprocess_data(directory_path_reprocessed, file_prefix_reprocessed, 16)
    data_new_pred = load_and_preprocess_data(directory_path_predictions_reprocessed, 'predicted_reprocessed_chunk_results__', 16)


    #X, y = get_features_targets(data)
    X, y = get_features_targets_alpha(data_new_pred)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #model = train_polynomial_regression(X, y, X_test, y_test) #Decent

    #model = train_random_forest(X, y, X_test, y_test) #Good
    #model = train_linear_regression(X, y, X_test, y_test) #Bad
    model = train_neural_network(X, y, X_test, y_test) #Best

    #predict_and_save(directory_path_reprocessed, file_prefix_reprocessed, 16, model, directory_path_predictions_reprocessed)









    #Pareto stuff
    """

    pareto_df = make_pareto_front(data)
    pareto_df_pred = make_pareto_front_pred(data_pred)

    pareto_df_sorted = pareto_df.sort_values(by='average_energy', ascending=True)
    pareto_df_pred_sorted = pareto_df_pred.sort_values(by='predicted_energy', ascending=True)


    pareto_df_sorted.to_csv('pareto_df_sorted.csv', index=False)


    pareto_df_pred_sorted.to_csv('pareto_df_pred_sorted.csv', index=False)

 
    plot_pareto_fronts(pareto_df, pareto_df_pred)


    # Assuming actual_pareto_df and predicted_pareto_df are your dataframes
    plt.figure(figsize=(10, 6))

    # Actual Pareto Front
    plt.scatter(pareto_df['average_velocity'], pareto_df['average_energy'],
                color='blue', alpha=0.5, label='Actual')

    # Predicted Pareto Front
    plt.scatter(pareto_df_pred['average_velocity'], pareto_df_pred['predicted_energy'] if 'predicted_energy' in pareto_df_pred else pareto_df_pred['average_energy'],
                color='red', alpha=0.5, label='Predicted')

    plt.xlabel('Average Velocity')
    plt.ylabel('Average Energy')
    plt.title('Overlap of Actual and Predicted Pareto Fronts')
    plt.legend()
    plt.gca().invert_yaxis()  # If applicable, to visualize lower energy values as better
    plt.show()


    plt.figure(figsize=(10, 6))

   # Example: alpha_h vs. average_velocity
    plt.scatter(pareto_df['alpha_h'], pareto_df['average_energy'],
                color='blue', alpha=0.5, label='Actual alpha_h')
    plt.scatter(pareto_df_pred['alpha_h'], pareto_df_pred['predicted_energy'],
                color='red', alpha=0.5, label='Predicted alpha_h')

    plt.xlabel('alpha_h')
    plt.ylabel('Average Energy')
    plt.title('Alpha_h Influence on Energy')
    plt.legend()
    plt.show()

    # Example: omega_h vs. average_energy
    plt.scatter(pareto_df['omega_h'], pareto_df['average_energy'],
                color='blue', alpha=0.5, label='Actual omega_h')
    plt.scatter(pareto_df_pred['omega_h'], pareto_df_pred['predicted_energy'],
                color='red', alpha=0.5, label='Predicted omega_h')

    plt.xlabel('omega_h')
    plt.ylabel('Average Energy')
    plt.title('Omega_h Influence on Energy')
    plt.legend()
    plt.show()


    # Example: delta_h vs. average_energy
    plt.scatter(pareto_df['delta_h'], pareto_df['average_energy'],
                color='blue', alpha=0.5, label='Actual delta_h')
    plt.scatter(pareto_df_pred['delta_h'], pareto_df_pred['predicted_energy'],
                color='red', alpha=0.5, label='Predicted delta_h')

    plt.xlabel('delta_h')
    plt.ylabel('Average Energy')
    plt.title('delta_h Influence on Energy')
    plt.legend()
    plt.show()

    """


















