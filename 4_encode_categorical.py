import pandas as pd
import re
import ast

def encode_column(df, column_name, prefix=None):
    # Perform one-hot encoding with the specified prefix
    encoded_df = pd.get_dummies(df[column_name].apply(pd.Series).stack(), prefix=prefix).groupby(level=0).sum()

    # Concatenate the encoded DataFrame with the original DataFrame
    df_encoded = pd.concat([encoded_df], axis=1)

    # Get the number of columns that were encoded
    num_encoded_columns = encoded_df.shape[1]

    return df_encoded, num_encoded_columns

data = pd.read_csv(r'Dataframe\4_ready_to_encode_categorical.csv')

print(f"Shape of original data set is {data.shape}")

categorical_features = [
    'MPAA Rating',
    'Keywords',
    'Source','Production method','Creative type','Countries'
]

df = pd.DataFrame()

for column in categorical_features:
    data[column] = data[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    encoded_df, num_encoded_columns = encode_column(data, column, prefix=column)
    print(f"\nEncoding column: {column}, with {num_encoded_columns} encoded columns")
    print("DataFrame after encoding:")
    print(encoded_df.head(5))
    df = pd.concat([df,encoded_df], axis=1)

df.to_csv(r'Dataframe\5_encoded_categorical.csv')
