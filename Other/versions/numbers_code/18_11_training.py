import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import r2_score, make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import time
import matplotlib.pyplot as plt

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
lasso_avg = [[],[]]
ridge_avg = [[],[]]
elastic_avg = [[],[]]
rf_avg = [[],[]]
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
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    lasso = Lasso(max_iter=num_iteration)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring=scorer)
    grid_search.fit(X_train, y_train)
    best_lasso_model = grid_search.best_estimator_
    lasso_predictions = best_lasso_model.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_predictions)
    lasso_rmse = mean_squared_error(y_test,lasso_predictions, squared=False)
    print(f"Lasso r2 on test set: {lasso_r2}")
    lasso_avg[0].append(lasso_r2)
    lasso_avg[1].append(lasso_rmse)

    ridge = Ridge(max_iter=num_iteration)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring=scorer)
    ridge_grid_search.fit(X_train, y_train)
    best_ridge_model = ridge_grid_search.best_estimator_
    ridge_predictions = best_ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_predictions)
    ridge_rmse = mean_squared_error(y_test,ridge_predictions, squared=False)
    print(f"Ridge r2 on test set: {ridge_r2}")
    ridge_avg[0].append(ridge_r2)
    ridge_avg[1].append(ridge_rmse)

    elastic_net = ElasticNet(max_iter=num_iteration)
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }
    elastic_net_grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring=scorer)
    elastic_net_grid_search.fit(X_train, y_train)
    best_elastic_net_model = elastic_net_grid_search.best_estimator_
    elastic_net_predictions = best_elastic_net_model.predict(X_test)
    elastic_r2 = r2_score(y_test, elastic_net_predictions)
    elastic_rmse = mean_squared_error(y_test,elastic_net_predictions, squared=False)
    print(f"Elastic r2 on test set: {elastic_r2}")
    elastic_avg[0].append(elastic_r2)
    elastic_avg[1].append(elastic_rmse)

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
    rf_r2 = r2_score(y_test, elastic_net_predictions)
    rf_rmse = mean_squared_error(y_test,rf_predictions, squared=False)
    print(f"RF r2 on test set: {rf_r2}")
    rf_avg[0].append(rf_r2)
    rf_avg[1].append(rf_rmse)

    save_to_csv(value, y_test=y_test,lasso_predictions=lasso_predictions,
                ridge_predictions=ridge_predictions,elastic_net_predictions=elastic_net_predictions,rf_predictions=rf_predictions)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("\nTraining summary")
print(f"Elapsed Time: {elapsed_time//60} minutes : {elapsed_time % 60} seconds")
print("")
print(f"Lasso average r2 {sum(lasso_avg[0])/len(lasso_avg[0])}")
print(f"Ridge average r2 {sum(ridge_avg[0])/len(ridge_avg[0])}")
print(f"Elastic net average r2 {sum(elastic_avg[0])/len(elastic_avg[0])}")
print(f"Random Forest average r2 {sum(rf_avg[0])/len(rf_avg[0])}")
print("")
print(f"Lasso average RMSE {sum(lasso_avg[1])/len(lasso_avg[1])}")
print(f"Ridge average RMSE {sum(ridge_avg[1])/len(ridge_avg[1])}")
print(f"Elastic net average RMSE {sum(elastic_avg[1])/len(elastic_avg[1])}")
print(f"Random Forest average RMSE {sum(rf_avg[1])/len(rf_avg[1])}")

# Plotting four columns against 'x' on the same graph
plt.plot(list_month, lasso_avg[0], label='Lasso')
plt.plot(list_month, ridge_avg[0], label='Ridge')
plt.plot(list_month, elastic_avg[0], label='Elastic Net')
plt.plot(list_month, rf_avg[0], label='Random Forest')

plt.title('Model R2 values against month')
plt.xlabel('Month')
plt.ylabel('R2 Value')
plt.legend()
plt.show()