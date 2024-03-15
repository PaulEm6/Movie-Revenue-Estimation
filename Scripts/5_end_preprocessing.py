import pandas as pd

data_numerical = pd.read_csv(r'Dataframe\2_scaled_numerical.csv')
data_categorical = pd.read_csv(r'Dataframe\5_encoded_categorical.csv')
data_original = pd.read_csv(r'Dataframe\0_movies1930_2022.csv')

categorical_features = [
    'MPAA Rating',
    'Keywords',
    'Source','Production method','Creative type','Countries'
]

for column in categorical_features:
    data_numerical = data_numerical.drop(column, axis=1)

data_categorical = data_categorical.drop(data_categorical.columns[0], axis=1)

data = pd.concat([data_numerical, data_categorical], axis=1)

data.to_csv(r'Dataframe\6_training_data.csv')