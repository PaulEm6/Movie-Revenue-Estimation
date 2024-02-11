import pandas as pd
from sklearn.preprocessing import StandardScaler


#Get dataset
data =pd.read_csv(r'Dataframe\tmdb_5000_movies.csv')

###################
# FEATURE SELECTION
###################

print("\n1. Feature selection")

#ALL FEATURES:
#budget,genres,homepage,id,keywords,original_language,original_title,overview,popularity,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,vote_average,vote_count
#,,,,,,overview,,,,,,,,,,,
#You can not train on domestic and international Box office
#Companies has too many errors
#Maybe consider them as estimations?

features = [
    'title','revenue', 'release_date', #Reference, Grouping, Target
    'budget','runtime',#Numerical features
    'genres', 'keywords','production_countries','production_companies', 'original_language', #categorical featuers
]

data = data[features]

# Convert 'release_date' to datetime format
data['release_date'] = pd.to_datetime(data['release_date'])

# Extract date, month, and year into separate columns
data['Day'] = data['release_date'].dt.day
data['Month'] = data['release_date'].dt.month
data['Year'] = data['release_date'].dt.year

# Drop the original 'release_date' column if needed
data = data.drop(columns=['release_date'])

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
#We do NOT perform scaling on the grouping of the model

numerical_features = [
    'budget','runtime', 'Day','Year'
]

#Statistics of numerical data in dataset
print("\nStatistics of numerical dataset before cleaning")
print(data[numerical_features].describe())

#Removing rows where values are too high or too low
data = data[~(data['budget'] < 1e+3)]
data = data[~(data['runtime'] < 1e+1)]

print("\nStatistics of numerical dataset after cleaning")
print(data[numerical_features].describe())

print("\nStatistics of target")
print(data['revenue'].describe())
data = data[~(data['revenue'] < 1e+3)]
print(data['revenue'].describe())


print("\nShape of dataset after cleaning: " + str(data.shape))

########################################################
# DATA PREPROCESSING: ENCODING CATEGORICAL DATA AND DATE
#######################################################

#Perform one hot encoding on the categorical data

print("\n4. Encoding categorical data")

categorical_features = [
    'genres', 'keywords','production_countries','production_companies', 'original_language'
]

for feature in categorical_features:
    unique_values_count = data[feature].nunique()
    print(f"Number of unique values in '{feature}': {unique_values_count}")

#One hot encoding of each categorical feature
data = pd.get_dummies(data, columns=categorical_features, prefix='columns_to_encode')

#Shape of dataset after onehot encoding
print("\nShape of dataset after one hot encoding: " + str(data.shape))

######################################
# DATA PREPROCESSING: FEATURE SCALING 
######################################

#Perform one hot encoding on numerical columns

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