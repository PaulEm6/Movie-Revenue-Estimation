import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read data from CSV file into a DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
data = pd.read_csv('Dataframe\dataset.csv')

features = [
    'Title','Month','Total adjusted B.O', #Reference, Grouping, Target
    'Year','Adjusted budget','Duration', #Numerical features
    'Keywords', #categorical featuers
     #'MPAA Rating','Keywords','Source','Production method','Creative type','Countries', #categorical featuers

    'Action','Adventure','Animation','Comedy','Crime','Documentary', #Binary features
    'Drama','Family','Fantasy','History','Horror','Music','Mystery', #Binary features
    'Romance','Science Fiction','TV Movie','Thriller','War','Western' #Binary features
]

target = 'Total adjusted B.O'

numerical_features = [
    'Year','Adjusted budget','Duration'
]


# Plotting using Matplotlib
plt.figure(figsize=(12, 6))

# Plotting for feature1
plt.subplot(1, 3, 1)
plt.scatter(data[numerical_features[0]], data[target])
plt.title(f'Relationship between {numerical_features[0]} and {target}')
plt.xlabel(f'{numerical_features[0]}')
plt.ylabel(f'{target}')

# Plotting for feature2
plt.subplot(1, 3, 2)
plt.scatter(data[numerical_features[1]], data[target])
plt.title(f'Relationship between {numerical_features[1]} and {target}')
plt.xlabel(f'{numerical_features[1]}')
plt.ylabel(f'{target}')

# Plotting for feature3
plt.subplot(1, 3, 3)
plt.scatter(data[numerical_features[2]], data[target])
plt.title(f'Relationship between {numerical_features[2]} and {target}')
plt.xlabel(f'{numerical_features[2]}')
plt.ylabel(f'{target}')

plt.tight_layout()
plt.show()

# Plotting using Seaborn
plt.figure(figsize=(12, 6))

# Plotting for feature1
plt.subplot(1, 3, 1)
sns.scatterplot(x=f'{numerical_features[0]}', y=target, data=data)
plt.title(f'Relationship between {numerical_features[0]} and {target}')

# Plotting for feature2
plt.subplot(1, 3, 2)
sns.scatterplot(x=numerical_features[1], y=target, data=data)
plt.title(f'Relationship between {numerical_features[1]} and {target}')

# Plotting for feature2
plt.subplot(1, 3, 3)
sns.scatterplot(x=numerical_features[2], y=target, data=data)
plt.title(f'Relationship between {numerical_features[2]} and {target}')

plt.tight_layout()
plt.show()
