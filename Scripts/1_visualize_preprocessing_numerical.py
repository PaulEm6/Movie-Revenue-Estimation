import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

budget = 'Original budget'
box_office = 'Total original B.O'

# Read data from CSV file into a DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
data = pd.read_csv(r'Dataframe\1_cleaned_numerical.csv')

features = [
    'Title','Month',box_office, #Reference, Grouping, Target
    'Year',budget,'Duration', #Numerical features
    'MPAA Rating','Keywords','Source','Production method','Creative type','Countries', #categorical featuers

    'Action','Adventure','Animation','Comedy','Crime','Documentary', #Binary features
    'Drama','Family','Fantasy','History','Horror','Music','Mystery', #Binary features
    'Romance','Science Fiction','TV Movie','Thriller','War','Western' #Binary features
]

target = box_office

numerical_features = [
    'Year',budget,'Duration'
]


# Plotting using Matplotlib
plt.figure(figsize=(12, 6))
counter = 0

for column in numerical_features:
    # Plotting for feature1
    plt.subplot(1, 3, counter+1)
    
    plt.scatter(data[numerical_features[counter]], data[target])
    

    # Extract the independent variable (X) and dependent variable (y)
    X = data[[column]]
    y = data[target]

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)
    # Make predictions using the fitted model
    predictions = model.predict(X)

    # Plot the original data and the linear regression line
    plt.scatter(X, y, label='Original Data')
    plt.plot(X, predictions, color='red', label='Linear Regression Line')
    plt.xlabel(column)
    plt.ylabel(target)
    plt.title('Linear Regression Model')
    plt.legend()
    counter = counter + 1
    correlation_coefficient = data[column].corr(data[target])
    # Print the slope (coefficient) and y-intercept of the linear regression model
    print(f"\nColumn: {column} Slope (Coefficient): {model.coef_[0]:.2f} ")
    print(f"Y-Intercept: {model.intercept_:.2f}")

plt.tight_layout()
plt.show()

