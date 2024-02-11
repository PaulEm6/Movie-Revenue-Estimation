import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
import re
import ast  # Library for literal_eval
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

def repl_quotes(m):
  preq = m.group(1)
  qbody = m.group(2)
  qbody = re.sub(r'"', r"'", qbody)
  return preq + '"' + qbody + '"'

def to_json(s):
  safe = s.replace("'", '"')
  safe = re.sub(r'("[\s\w]*)"([\s\w]*")',r"\1'\2", safe)  # O'Brien
  safe = re.sub( r'([:\[,{]\s*)"(.*?)"(?=\s*[:,\]}])', repl_quotes, safe ) # Alex "Nickname" Schittko
  safe = safe.replace("None", 'null')
  safe = safe.replace("\\'", "'")
  safe = safe.replace("\\x92", "'")
  safe = safe.replace("\\xa0", "-")
  safe = safe.replace("\\xad", "-")
  #print(safe)
  try:
    cast_json = json.loads(safe)
  except:
    print("to_json() failed for string")
    print(safe)

  return cast_json, safe

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

    keywords_df.to_csv(rf'numbers_code\clean\categorical_data\{column_name}_occurrences.csv', index=False)

    return unique_keywords, keyword_occurrences, top_keywords_occurrences

def remove_keywords_containing_phrase(dataframe, column_name, phrase):
    modified_dataframe = dataframe.copy()
    for index, row in modified_dataframe.iterrows():
        modified_dataframe.at[index, column_name] = [keyword for keyword in row[column_name] if phrase not in keyword]
    return modified_dataframe


data = pd.read_csv('Dataframe\dataset.csv')

print(f"Shape of original data set is {data.shape}")

categorical_features = [
    'MPAA Rating',
    'Keywords',
    'Source','Production method','Creative type','Countries'
]

#Proper format of as strings
for column in categorical_features:
    if column != 'Keywords' and column != 'Countries':
        data[column] = data[column].apply(lambda x: "['{}']".format(x))
                                          
for column in categorical_features:

    # Create an empty DataFrame with the desired columns
    result_df = pd.DataFrame(columns=['Keywords_cleaned', 'safe_column'])

    # Assuming 'Keywords' is the column you want to convert to JSON in the original DataFrame
    result_df[['Keywords_cleaned', 'safe_column']] = data[column].apply(to_json).apply(pd.Series)

    data[column] = result_df['Keywords_cleaned']

    if column == 'Keywords':

        # Specify the phrase to check for
        phrase_to_remove = 'Filmed in'

        # Remove keywords containing the specified phrase
        modified_df = remove_keywords_containing_phrase(data, column, phrase_to_remove)
        
        data[column] = modified_df[column]

    top_n = 5
    unique_keywords, keyword_occurrences, top_keywords_occurrences = analyze_categorical_data(data, column, top_n)

    # Print the results
    print(f"\nColumn:{column} Total number of unique categorical values: {len(unique_keywords)}")

    # Print the top N keywords and their occurrences
    if top_keywords_occurrences is not None:
        print(f"Top {top_n} categorical values and their occurrences:")
        for keyword, occurrences in top_keywords_occurrences.items():
            print(f"{keyword}: {occurrences} occurrences")

data[categorical_features].to_csv('numbers_code\clean\cleaned_categorical.csv')


