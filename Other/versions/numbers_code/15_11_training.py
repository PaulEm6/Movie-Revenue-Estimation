import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import r2_score, make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import time

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
    dataframe.to_csv(f"Dataframe/model_month/model_month_{month}.csv", index=False)

# Record the start time
start_time = time.time()

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('Dataframe/dataset.csv')

#Looking at shape of dataset
print("\nShape of dataset " + str(df.shape))

# Specify the column you want to use for grouping
grouping_column = 'Month'
target_column = 'Total adjusted B.O'

# Get unique values in the specified column for grouping
unique_values = df[grouping_column].unique()
list_month = sorted(unique_values)

#Lists for average r2
lasso_avg = []
ridge_avg = []
elastic_avg = []
rf_avg = []
num_iteration = 100000

# Iterate over unique values and train a Random Forest model for each value
for value in list_month:
    
    # Filter the dataset for the current value
    subset_data = df[df[grouping_column] == value]

    # Use R-squared as the scoring scorer   scorer = make_scorer(r2_score)
    scorer = make_scorer(r2_score)

    print(f"\nTraining for month {value}, number of movies: {subset_data.shape[0]}")

    # Split the data into features (X) and target variable (y)
    #We remove the grouping column, target column and title from training set
    X = subset_data.drop(columns=[grouping_column, target_column, 'Title'])  
    y = subset_data[target_column]  # Adjust 'target_column'

    # Split the data into training and testing sets, apply PCA for dimensionality reduction
    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    lasso = Lasso(max_iter=num_iteration)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring=scorer)
    grid_search.fit(X_train, y_train)
    best_lasso_model = grid_search.best_estimator_
    lasso_predictions = best_lasso_model.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_predictions)
    print(f"Lasso r2 on test set: {lasso_r2}")
    lasso_avg.append(lasso_r2)

    ridge = Ridge(max_iter=num_iteration)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring=scorer)
    ridge_grid_search.fit(X_train, y_train)
    best_ridge_model = ridge_grid_search.best_estimator_
    ridge_predictions = best_ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_predictions)
    print(f"Ridge r2 on test set: {ridge_r2}")
    ridge_avg.append(ridge_r2)

    elastic_net = ElasticNet(max_iter=num_iteration)
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }
    elastic_net_grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring=scorer)
    elastic_net_grid_search.fit(X_train, y_train)
    best_elastic_net_model = elastic_net_grid_search.best_estimator_
    elastic_net_predictions = best_elastic_net_model.predict(X_test)
    elastic_net_r2 = r2_score(y_test, elastic_net_predictions)
    print(f"Elastic Net r2 on test set: {elastic_net_r2}")
    elastic_avg.append(elastic_net_r2)

    rf_regressor = RandomForestRegressor()
    n_estimators = [3, 30, 300]
    param_grid = {
        'n_estimators': n_estimators
    }
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring=scorer)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    r2_scorer = make_scorer(lambda y, y_pred: np.sqrt(r2_score(y, y_pred)), greater_is_better=False)
    best_rf_regressor = RandomForestRegressor(**best_params)
    best_rf_regressor.fit(X_train, y_train)
    rf_predictions = best_rf_regressor.predict(X_test)
    rf_r2 = r2_score(y_test, rf_predictions)
    print(f"Random forest r2 on test set: {rf_r2}")
    rf_avg.append(rf_r2)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("\nTraining summary")
print(f"Elapsed Time: {elapsed_time//60} minutes : {elapsed_time % 60} seconds")
print(f"Lasso average r2 {sum(lasso_avg)/len(lasso_avg)}")
print(f"Ridge average r2 {sum(ridge_avg)/len(ridge_avg)}")
print(f"Elastic net average r2 {sum(elastic_avg)/len(elastic_avg)}")
print(f"Random Forest average r2 {sum(rf_avg)/len(rf_avg)}")

