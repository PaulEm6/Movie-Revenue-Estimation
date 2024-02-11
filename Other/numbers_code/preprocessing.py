import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import text
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

def preprocess_keyword_column(keyword_column, min_frequency=2):
    processed_column = []

    for keyword_list in keyword_column:
        # Convert the list to lowercase
        keyword_list = [keyword.lower() for keyword in keyword_list]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        keyword_list = [keyword for keyword in keyword_list if keyword not in stop_words]

        # Calculate keyword frequencies
        keyword_counts = Counter(keyword_list)

        # Remove keywords with frequency below the specified threshold
        keyword_list = [keyword for keyword, count in keyword_counts.items() if count >= min_frequency]

        # Convert the processed list to a tuple
        processed_column.append(tuple(keyword_list))

    return processed_column

#Get dataset
data =pd.read_csv('Dataframe\movies1930_2022.csv')

###################
# FEATURE SELECTION
###################

print("\n1. Feature selection")

#ALL FEATURES:
#Title,Year,Month,MPAA Rating,Original domestic B.O,Original international B.O,Adjusted domestic B.O,Adjusted international B.O,Original budget,Adjusted budget,Duration,Keywords,Source,Production method,Creative type,Companies,Countries,Languages,Genres,Action,Adventure,Animation,Comedy,Crime,Documentary,Drama,Family,Fantasy,History,Horror,Music,Mystery,Romance,Science Fiction,TV Movie,Thriller,War,Western,Total original B.O,Total adjusted B.O
#You can not train on domestic and international Box office
#Companies has too many errors
#Maybe consider them as estimations?

features = [
    'Title','Month','Total adjusted B.O', #Reference, Grouping, Target
    'Year','Adjusted budget','Duration', #Numerical features
    'MPAA Rating','Keywords','Source','Production method','Creative type','Countries', #categorical featuers

    'Action','Adventure','Animation','Comedy','Crime','Documentary', #Binary features
    'Drama','Family','Fantasy','History','Horror','Music','Mystery', #Binary features
    'Romance','Science Fiction','TV Movie','Thriller','War','Western' #Binary features
]

data = data[features]

print("Original dataset with feature selection: " + str(data.shape))

#############################################
# DATA PREPROCESSING: REMOVING MISSING VALUES 
#############################################

print("\n2. Removing missing values")

# Check for missing values in the entire dataset
print("\nMissing values for each column")
print(data.isnull().sum())

#We remove all the missing values, at most 1896 rows will be lost
data.dropna(inplace=True)

# Check for missing values in the entire dataset
print("\nMissing values for each column after removing all missing values")
print(data.isnull().sum())

print("\nShape of dataset after handling missing values is: " + str(data.shape))

########################################
# DATA PREPROCESSING: HANDLING OUTLIERS 
########################################

print("\n3. Handling outliers")

#First we look at outliers of numerical values
#We do NOT perform scaling on the target of the model
numerical_features = [
    'Year','Adjusted budget','Duration'
]

#Statistics of numerical data in dataset
print("\nStatistics of numerical dataset before cleaning")
print(data[numerical_features].describe())

#Removing rows where values are too high or too low
data = data[~(data['Adjusted budget'] < 1e+3)]
data = data[~(data['Adjusted budget'] > 1e+9)]
data = data[~(data['Total adjusted B.O'] > 3e+9)]
data = data[~(data['Duration'] > 1e+5)]

#Removing zero values in numerical features
data = data[~(data[numerical_features] == 0).any(axis=1)]

print("\nStatistics of numerical dataset after cleaning")
print(data[numerical_features].describe())

print("\nStatistics of target")
print(data['Total adjusted B.O'].describe())

print("\nShape of dataset after cleaning: " + str(data.shape))

###############################################
# DATA PREPROCESSING: ENCODING CATEGORICAL DATA 
###############################################

#Perform one hot encoding on the categorical data

print("\n4. Encoding categorical data")

categorical_features = [
    'MPAA Rating','Keywords','Source','Production method','Creative type','Countries'
]


for feature in categorical_features:
    unique_values_count = data[feature].nunique()
    print(f"Number of unique values in '{feature}': {unique_values_count}")



######################################
# DATA PREPROCESSING: FEATURE SCALING 
######################################

#Perform one hot encoding on the column "keywords"

print("\n5. Feature scaling")

print("\nStatistics of numerical dataset before scaling")
print(data[numerical_features].describe())

# Standard Scaling
standard_scaler = StandardScaler()
data[numerical_features] = standard_scaler.fit_transform(data[numerical_features])

print("\nStatistics of numerical dataset after scaling")
print(data[numerical_features].describe())

print("\nShape of dataset: " + str(data.shape))

##################################################
# SAVING DATAFRAME INTO NEW CSV FILE FOR TRAINING
##################################################

print("\n6. Saving dataframe into new csv file")

print("\nFinal shape of dataset: " + str(data.shape))

data.to_csv('Dataframe\dataset.csv', index=False)