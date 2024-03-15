import pandas as pd
import ast  # module for safely evaluating literal expressions from strings
from collections import Counter


def analyze_categorical_data(dataframe, column_name, top_n=None):

    # Assuming the column with categorical value is named 'keywords', and each cell contains a list of keywords
    all_keywords = [keyword for sublist in dataframe[column_name] for keyword in sublist]
    # Count the occurrences of each categorical value
    keyword_counts = Counter(all_keywords)
    # Get the unique keywords and their counts
    unique_keywords = list(keyword_counts.keys())
    keyword_occurrences = list(keyword_counts.values())

    # Find the top N most common keywords and their occurrences
    top_keywords_occurrences = None
    if top_n is not None:
        sorted_keywords = dict(keyword_counts.most_common(top_n))
        top_keywords_occurrences = {keyword: sorted_keywords[keyword] for keyword in sorted_keywords}

    # Create a DataFrame with the results
    keywords_df = pd.DataFrame(list(keyword_counts.items()), columns=[column_name, 'Occurrences'])
    # Sort the DataFrame in descending order based on occurrences
    keywords_df = keywords_df.sort_values(by='Occurrences', ascending=False)
    keywords_df.to_csv(rf'Dataframe\categorical_data\{column_name}_occurrences.csv', index=False)

    # Calculate the total occurrences
    total_occurrences = sum(keyword_counts.values())
    return total_occurrences, unique_keywords, keyword_occurrences, top_keywords_occurrences

def transform_keywords(dataframe, column_name, occurrence_threshold):
    # Convert string representations of lists to actual lists
    dataframe[column_name] = dataframe[column_name].apply(ast.literal_eval)

    # Step 1: Count the occurrences of each keyword
    keyword_counts = pd.Series([keyword for keywords in dataframe[column_name] for keyword in keywords]).value_counts()

    # Step 2: Identify keywords that don't meet the occurrence threshold
    keywords_to_replace = keyword_counts[keyword_counts < occurrence_threshold].index.tolist()

    # Step 3: Replace those keywords with the "other" keyword, avoiding duplication
    dataframe[column_name] = dataframe[column_name].apply(lambda keywords: list(set([f'Other_{column}' if keyword in keywords_to_replace else keyword for keyword in keywords])))

    return dataframe

data = pd.read_csv(r'Dataframe\3_cleaned_categorical.csv')

print(f"Shape of original data set is {data.shape}")

multiple_categorical = [
    'Keywords','Countries'
]

# Set the occurrence threshold
for column in multiple_categorical:
    occurrence_threshold = 50
    # Apply the method
    data[column] = transform_keywords(data, column, occurrence_threshold)[column]

    top_n = 20
    total_occurences, unique_keywords, keyword_occurrences, top_keywords_occurrences = analyze_categorical_data(data, column, top_n)
        
    # Print the results
    print(f"\nColumn: {column}")
    print(f"Total number of unique categorical values: {len(unique_keywords)}")
    print(f"Total number of occurences: {total_occurences}")
    # Print the top N keywords and their occurrences
    if top_keywords_occurrences is not None:
        print(f"Top {top_n} categorical values and their occurrences:")
        for keyword, occurrences in top_keywords_occurrences.items():
            print(f"{keyword}: {occurrences} occurrences")

data = data.drop(data.columns[0], axis=1)

data.to_csv(r'Dataframe\4_ready_to_encode_categorical.csv')
