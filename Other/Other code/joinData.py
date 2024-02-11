import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
def parse_list_string(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []

def getRowsWithMissingValuePerCol(colname):
    rows_with_missing_values = df[df[colname].isna()]
    print(f"Rows where {colname} has missing values (NaN):")
    print(rows_with_missing_values[["Title", "Year", colname]])

def getDistinctValues(colname):
    distinct_values = df[colname].unique()
    print(f"Distinct values in {colname} column:")
    print(distinct_values)

def displayColsWithMissingValues(df):
    columns_with_missing_values = df.columns[df.isna().any()]
    print("Columns with missing values:")
    print(columns_with_missing_values)

def displayNbMissingValues(df):
    print("Missing values:")
    missing_values_per_column = df.isna().sum()
    print(missing_values_per_column)

def predictCol(df, target, features):
    train_data = df.dropna(subset=[target])
    test_data = df[df[target].isna()]
    features = np.append(mlb.classes_, "Year")

    X_train = train_data[features]
    y_train = train_data[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Test if model is good
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    for fold, score in enumerate(scores, start=1):
        print(f'Fold {fold} for col {target}: Accuracy = {score:.4f}')
    average_accuracy = scores.mean()
    print(f'Average Accuracy for col {target}: {average_accuracy:.4f}')

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    X_test = test_data[features]
    predicted_values = model.predict(X_test)
    df.loc[df[target].isna(), target] = predicted_values
    return df

# Step 1: Concatenate all CSV files into one DataFrame
dfs = [pd.read_csv("movies" + str(year) + ".csv") for year in range(1930, 2022)]
df = pd.concat(dfs, ignore_index=True)

# Transform Genre to remove string aspect
df['Genres'] = df['Genres'].apply(parse_list_string)

# Handle missing values for MPAA and BO
df['MPAA Rating'] = df["MPAA Rating"].fillna("Not Rated")
columns_to_fill_with_zero = ['Original domestic B.O', 'Original international B.O',
                             'Adjusted domestic B.O', 'Adjusted international B.O']
df[columns_to_fill_with_zero] = df[columns_to_fill_with_zero].fillna(0)

# Discretize genre
mlb = MultiLabelBinarizer()
binary_genres = mlb.fit_transform(df['Genres'])
binary_df = pd.DataFrame(binary_genres, columns=mlb.classes_)
df = pd.concat([df, binary_df], axis=1)

# Fill Production method
features = np.append(mlb.classes_, "Year")
df = predictCol(df, "Production method", features)



# Remove cols with missing values
df = df.dropna(subset=['Original budget', "Duration", "Source", "Creative type"])

displayColsWithMissingValues(df)
displayNbMissingValues(df)

# Create new columns
df['Total original B.O'] = df['Original domestic B.O'] + df['Original international B.O']
df['Total adjusted B.O'] = df['Adjusted domestic B.O'] + df['Adjusted international B.O']


df = df.drop_duplicates(subset=['Title', 'Year'])
info = df.describe()



df.to_csv('movies1930_2022.csv', index=False)