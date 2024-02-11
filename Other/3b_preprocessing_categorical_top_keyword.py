import pandas as pd
from ast import literal_eval

# Read the CSV file into a DataFrame
file_path = 'Dataframe\cleaned_categorical.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Function to extract the first value from a list
def extract_first_value(lst):
    return [str(lst[0])]


column_name = 'Keywords'
# Apply the function to the specified column
top_keywords = df[column_name].apply(eval).apply(extract_first_value)

column_name = 'Countries'
# Apply the function to the specified column
top_countries = df[column_name].apply(eval).apply(extract_first_value)

data = pd.concat([top_keywords, top_countries, df['Total adjusted B.O']], axis=1)

multiple_categorical_features = [
    'Keywords',
    'Countries'
]

for column in multiple_categorical_features:
    # Replace 'your_column_name' with the actual name of the column you're interested in
    print(f"\nAnalzing column: {column}")
    column_counts = df[column].value_counts().head(10)
    # Display each unique value and its occurrence
    for value, count in column_counts.items():
        print(f"{value}: {count} occurrences")

    # Calculate the sum of all occurrences in the entire DataFrame
    total_occurrences = df[column].value_counts().sum()

    # Display the sum of total occurrences
    print(f"Total occurrences in the entire DataFrame: {total_occurrences}")

# Save the modified DataFrame back to a new CSV file or overwrite the existing one
data.to_csv(r'Dataframe\treated_categorical.csv')

