import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Function used to save the models performance for a given month
def save_to_csv(month, y_test, lasso_predictions, ridge_predictions, elastic_net_predictions, rf_predictions):
    data = {
        'Total adjusted box office': y_test,
        'Lasso predictions': lasso_predictions,
        'Ridge predictions': ridge_predictions,
        'Elastic net predictions': elastic_net_predictions,
        'Random Forest predictions': rf_predictions
           }
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(f"../model_month/model_month_{month}.csv", index=False)

# Record the start time
start_time = time.time()

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('../dataset.csv')

#Looking at shape of dataset
print("\nShape of dataset " + str(df.shape))

# Specify the column you want to use for grouping
grouping_column = 'Month'
target_column = 'Profitability'

# Get unique values in the specified column for grouping
unique_values = df[grouping_column].unique()
list_month = sorted(unique_values)

num_iteration = 100000

params = {
    "Lasso": {
        "model": Lasso(max_iter=num_iteration),
        "grid": {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        "predictions": []
    },
    "Ridge": {
        "model": Ridge(max_iter=num_iteration),
        "grid": {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        "predictions": []
    },
    "ElasticNet": {
        "model": ElasticNet(max_iter=num_iteration),
        "grid": {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        "predictions": []
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(),
        "grid": {
            'n_estimators': [3, 30, 300]
        },
        "predictions": []
    },
    "Poly": {
        "model": make_pipeline(PolynomialFeatures(), LinearRegression()),
        "grid": {
            'polynomialfeatures__degree': [1, 2, 3, 4, 5],  # Try different degrees
        },
        "predictions": []
    }
}

# Iterate over unique values and train a Random Forest model for each value
for value in list_month:
    
    # Filter the dataset for the current value
    subset_data = df[df[grouping_column] == value]

    print(f"\nTraining for month {value}, number of movies: {subset_data.shape[0]}")

    # Split the data into features (X) and target variable (y)
    #We remove the grouping column, target column and title from training set
    X = subset_data.drop(columns=[grouping_column, target_column, 'Title'])  
    y = subset_data[target_column]  # Adjust 'target_column'

    # Split the data into training and testing sets, apply PCA for dimensionality reduction
    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    for name in params:
        param = params[name]
        model = param["model"]
        grid = param["grid"]
        grid_search = GridSearchCV(model, grid, cv=10, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print(f"{name} RMSE on test set: {rmse}")
        param["predictions"].append(rmse)

    #save_to_csv(value, y_test=y_test, lasso_predictions=lasso_predictions,
    #ridge_predictions=ridge_predictions, elastic_net_predictions=elastic_net_predictions, rf_predictions=rf_predictions)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("\nTraining summary")
print(f"Elapsed Time: {elapsed_time//60} minutes : {elapsed_time % 60} seconds")
print(f"Lasso average RMSE {sum(params['Lasso']['predictions'])/len(params['Lasso']['predictions'])}")
print(f"Ridge average RMSE {sum(params['Ridge']['predictions'])/len(params['Ridge']['predictions'])}")
print(f"Elastic net average RMSE {sum(params['ElasticNet']['predictions'])/len(params['ElasticNet']['predictions'])}")
print(f"Random Forest average RMSE {sum(params['RandomForestRegressor']['predictions'])/len(params['RandomForestRegressor']['predictions'])}")

