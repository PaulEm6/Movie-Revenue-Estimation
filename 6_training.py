# pylint: disable-all

import os
import time
import pandas as pd
import joblib
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Set the LOKY_MAX_CPU_COUNT environment variable to the number of cores you want to use
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Replace '4' with the desired number of cores

budget = 'Original budget'
box_office = 'Total original B.O'

#Function used to save the models performance for a given month
def save_to_csv(name, y_test, predictions_test):
    data = {
        box_office : y_test,
        f'{name} predictions': predictions_test,
           }
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(rf"Dataframe\model_result\{name}.csv", index=False)

# Define a custom scorer for positive R2 score
def custom_r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    r2 = r2_score(y, y_pred)
    return r2

# Record the start time
start_time = time.time()

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv(r'Dataframe\6_training_data.csv')

# Specify the column you want to use for grouping
grouping_column = 'Month'
target_column = box_office

# Get unique values in the specified column for grouping
unique_values = df[grouping_column].unique()
list_month = sorted(unique_values)

num_iteration = 100000


Predictions_list = [[],[],[],[]]

params = {
    "KNN" : {
        "model": KNeighborsRegressor(),
        "grid": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
        },
        "rmse": [],
        "r2": [],
        "best_params": [],

    }
}

print(f"\nTraining, dataset shape: {df.shape}")

# Split the data into features (X) and target variable (y)
#We remove the grouping column, target column and title from training set
X = df.drop(columns=[ target_column, 'Title'])  
X = X.drop(df.columns[0], axis=1)

print(X.shape)
y = df[target_column]  # Adjust 'target_column'

# Split the data into training and testing sets, apply PCA for dimensionality reduction
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X)
mean_training = pca.mean_
components_training = pca.components_
joblib.dump(pca, f'pca_model.joblib')

# Assuming X and y are your feature matrix and target variable
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

for name in params:
    param = params[name]
    model = param["model"]
    grid = param["grid"]
    
    # Use X_train and y_train for training and X_val, y_val for validation
    grid_search = GridSearchCV(model, grid, cv=10, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = best_model.get_params()
    
    # Save the best model to a pickle file
    model_filename = f"Models\{name}_best_model.pkl"
    
    # pylint: disable=consider-using-with
    joblib.dump(best_model, model_filename)
    
    print(f"Saved {name} best model to {model_filename}")
    
    # Store the best model parameters
    param["best_params"] = best_params
    
    # Evaluate on the validation set
    predictions_val = best_model.predict(X_val)
    rmse_val = mean_squared_error(y_val, predictions_val, squared=False)
    r2_val = r2_score(y_val, predictions_val)
    
    # Evaluate on the test set
    predictions_test = best_model.predict(X_test)
    rmse_test = mean_squared_error(y_test, predictions_test, squared=False)
    r2_test = r2_score(y_test, predictions_test)
    
    print(f"\n{name} Best Model Parameters:")
    print(f"{best_params}")
    print(f"\n{name} Validation RMSE: {rmse_val}. Validation R2: {r2_val}")
    print(f"{name} Test RMSE: {rmse_test}. Test R2: {r2_test}")
    
    param["rmse"].append(rmse_test)  # Appending test set RMSE for comparison
    param["r2"].append(r2_test)      # Appending test set R2 for comparison

    save_to_csv(name, y_test, predictions_test=predictions_test)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
hours = elapsed_time // 3600
minutes = (elapsed_time % 3600) // 60
seconds = elapsed_time % 60

print("\nTraining summary")
print(f"Elapsed Time: {hours} hours : {minutes} minutes : {seconds} seconds")
# Creating a DataFrame
data = pd.DataFrame({'Real_Values': y_test, 'Predicted_Values': predictions_test})

# Displaying the DataFrame
print("\nReal vs Predicted Value")
print(data.head(5))
