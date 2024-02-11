import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import time

#Function used to save the models performance for a given month
def save_to_csv(month, y_test, ridge_predictions, elastic_net_predictions, rf_predictions):
    data = {
        'Total adjusted box office': y_test,
        'Ridge predictions': ridge_predictions,
        'Elastic net predictions': elastic_net_predictions,
        'Random Forest predictions': rf_predictions
           }
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(f"Dataframe/model_month/model_month_{month}.csv", index=False)

# Record the start time
start_time = time.time()

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('Dataframe/dataset.csv')

#Looking at shape of dataset
print("\nShape of dataset " + str(df.shape))

# Specify the column you want to use for grouping
grouping_column = 'Month'
target_column = 'revenue'

# Get unique values in the specified column for grouping
unique_values = df[grouping_column].unique()
list_month = sorted(unique_values)

#Lists for average rmse
lasso_avg = []
ridge_avg = []
elastic_avg = []
rf_avg = []

# Iterate over unique values and train a Random Forest model for each value
for value in list_month:
    
    # Filter the dataset for the current value
    subset_data = df[df[grouping_column] == value]

    print(f"\nTraining for month {value}, number of movies: {subset_data.shape[0]}")

    # Split the data into features (X) and target variable (y)
    #We remove the grouping column, target column and title from training set
    X = subset_data.drop(columns=[grouping_column, target_column, 'title'])  
    y = subset_data[target_column]  # Adjust 'target_column'

    # Split the data into training and testing sets, apply PCA for dimensionality reduction
    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    ridge_grid_search.fit(X_train, y_train)
    best_ridge_model = ridge_grid_search.best_estimator_
    ridge_predictions = best_ridge_model.predict(X_test)
    ridge_rmse = mean_squared_error(y_test, ridge_predictions, squared=False)
    print(f"Ridge RMSE on test set: {ridge_rmse}")
    ridge_avg.append(ridge_rmse)

    elastic_net = ElasticNet()
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }
    elastic_net_grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')
    elastic_net_grid_search.fit(X_train, y_train)
    best_elastic_net_model = elastic_net_grid_search.best_estimator_
    elastic_net_predictions = best_elastic_net_model.predict(X_test)
    elastic_net_rmse = mean_squared_error(y_test, elastic_net_predictions, squared=False)
    print(f"Elastic Net RMSE on test set: {elastic_net_rmse}")
    elastic_avg.append(elastic_net_rmse)

    rf_regressor = RandomForestRegressor()
    n_estimators = [3, 30, 300]
    param_grid = {
        'n_estimators': n_estimators
    }
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)
    best_rf_regressor = RandomForestRegressor(**best_params)
    best_rf_regressor.fit(X_train, y_train)
    rf_predictions = best_rf_regressor.predict(X_test)
    rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
    print(f"Random forest RMSE on test set: {rf_rmse}")
    rf_avg.append(rf_rmse)

    save_to_csv(value, y_test=y_test,
    ridge_predictions=ridge_predictions, elastic_net_predictions=elastic_net_predictions, rf_predictions=rf_predictions)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("\nTraining summary")
print(f"Elapsed Time: {elapsed_time//60} minutes : {elapsed_time % 60} seconds")
#print(f"Lasso average RMSE {sum(lasso_avg)/len(lasso_avg)}")
print(f"Ridge average RMSE {sum(ridge_avg)/len(ridge_avg)}")
print(f"Elastic net average RMSE {sum(elastic_avg)/len(elastic_avg)}")
print(f"Random Forest average RMSE {sum(rf_avg)/len(rf_avg)}")

