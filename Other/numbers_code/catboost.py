import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load data from CSV
file_path = 'Dataframe\dataset.csv'
df = pd.read_csv(file_path)

# Identify the grouping column
group_column = 'Group'

# Get unique values in the grouping column
unique_groups = df[group_column].unique()

# Initialize an empty dictionary to store models
group_models = {}

# Loop through each unique group
for group_value in unique_groups:
    # Select data for the current group
    group_data = df[df[group_column] == group_value]

    # Split the data into features and target
    X = group_data.drop('Target', axis=1)
    y = group_data['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train a CatBoost model for each group
    model = make_pipeline(StandardScaler(), CatBoostRegressor(random_state=42, verbose=0))
    model.fit(X_train, y_train)

    # Store the trained model in the dictionary
    group_models[group_value] = model

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model for each group
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for Group {group_value}: {mse}")

# Now, group_models dictionary contains trained CatBoost models for each unique value in the 'Group' column.
# You can use these models for making predictions on new data corresponding to each group.
