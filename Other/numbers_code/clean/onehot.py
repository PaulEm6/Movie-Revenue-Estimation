import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def one_hot_encode_column(df, column_name, min_frequency=1):
    # Initialize CountVectorizer with a custom tokenizer
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)

    # Fit and transform the specified column
    X = vectorizer.fit_transform(df[column_name])

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Find indices of features that meet the minimum frequency threshold
    selected_indices = [i for i, feature in enumerate(feature_names) if X[:, i].sum() >= min_frequency]

    # Filter the feature names and transform the original column
    selected_features = [feature_names[i] for i in selected_indices]
    X_selected = X[:, selected_indices]

    # Create a new DataFrame with the one-hot encoded columns
    df_encoded = pd.DataFrame(X_selected.toarray(), columns=selected_features)

    # Concatenate the new DataFrame with the original one
    df = pd.concat([df, df_encoded], axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df

# Example usage:
data = {'text_column': ['apple banana apple', 'orange banana', 'apple pear', 'orange apple banana'],
        'text_column_2': ['apple banana apple', 'orange banana', 'apple pear', 'orange apple banana']}
df = pd.DataFrame(data)

# Apply the one_hot_encode_column function to the 'text_column' with a minimum frequency of 2
df_encoded = one_hot_encode_column(df, 'text_column', min_frequency=3)

# Display the result
print(df_encoded)
