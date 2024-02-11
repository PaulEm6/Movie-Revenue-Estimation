
import pandas as pd
import ast  # Library for literal_eval

def remove_quotes_around_brackets(keyword_list):
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal (list)
        keyword_list = ast.literal_eval(keyword_list)
    except (SyntaxError, ValueError):
        # Handle the case where literal_eval fails (e.g., invalid syntax)
        return keyword_list

    # Return the list of keywords
    return keyword_list

# Example DataFrame
data = {'keywords_column': ['["apple", "orange", "banana"]',
                            '["apple"]',
                            '["orange", "grape"]',
                            '["banana", "apple", "orange"]']}
df = pd.DataFrame(data)

# Apply the remove_quotes_around_brackets method to the 'keywords_column'
df['keywords_column'] = df['keywords_column'].apply(remove_quotes_around_brackets)

# Display the DataFrame after removing quotes around brackets
print(df)