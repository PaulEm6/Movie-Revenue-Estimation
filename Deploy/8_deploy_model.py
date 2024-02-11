import os
import json 
import joblib
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set the LOKY_MAX_CPU_COUNT environment variable to the number of cores you want to use
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Replace '4' with the desired number of cores

def load_model_from_joblib(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        print(f"Error loading the model from {file_path}: {e}")
        return None

def encode_categorical_features(true_features, json_filename='Deploy\column_names.json'):
    # Load all_features from the JSON file
    with open(json_filename, 'r') as json_file:
        all_features = json.load(json_file)

    encoded_vector = [1 if feature in true_features else 0 for feature in all_features]
    return np.array(encoded_vector)

def scale_numerical_inputs(numerical_inputs):
    # Example scaled parameters
    scaled_parameters = {
        'Year': {'mean': 2001.8869670763163, 'std': 14.084593842775673},
        'Original budget': {'mean': 34111136.278258234, 'std': 53715920.75652778},
        'Duration': {'mean': 108.96904870277652, 'std': 20.294769384770788},
    }

    scaled_values = []

    for feature in numerical_inputs:
        mean = scaled_parameters[feature]['mean']
        std = scaled_parameters[feature]['std']

        # Scale the numerical input using the local mean and std
        scaled_value = (numerical_inputs[feature] - mean) / std
        scaled_values.append(scaled_value)

    # Convert the scaled values to a NumPy array
    scaled_array = np.array(scaled_values)

    return scaled_array

def plot_box_office_returns(box_office_values):
    # Assuming you have 12 months in a year
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Check if the length of the input list matches the number of months
    if len(box_office_values) != len(months):
        raise ValueError("Number of values should match the number of months (12).")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(months, box_office_values, marker='o', linestyle='-')
    plt.title("Box Office Returns Over the Year")
    plt.xlabel("Month")
    plt.ylabel("Total Box Office Return ($)")
    plt.grid(True)
    #plt.show()

def predict(year: int, budget: int, duration: int, Genres: [], MPAA_rating: [], Keywords: [], Source: [], Production_Method: [], Creative_type: [], Countries: []):
    
    #Numerical features
    numerical_features = {
        'Year': year,
        'Original budget': budget,
        'Duration': duration,
    }
    scaled_numerical = scale_numerical_inputs(numerical_features)
    
    #print("\nOriginal Numerical Inputs:", numerical_features)
    #print("Scaled Inputs:", scaled_numerical)

    #Categorical features
    categoriacl_features = MPAA_rating+Keywords+Source+Production_Method+Creative_type+Countries+Genres
    encoded_categorical = encode_categorical_features(categoriacl_features)
    
    #print("\nEncoded categorical: ", encoded_categorical)

    box_office_list = []

    for i in range(1,13):

        #Month
        month_vector = np.array([[i]])

        # Reshape the vectors if needed (e.g., from (3,) to (3,1))
        encoded_categorical = encoded_categorical.reshape(1, -1)
        scaled_numerical = scaled_numerical.reshape(1, -1)
        month_vector = month_vector.reshape(1,-1)

        #Input Vector
        input_vector = np.concatenate((month_vector, scaled_numerical, encoded_categorical), axis=1)
        print(f"Input vector shape {input_vector.shape}")

        #Applying PCA
        pca = joblib.load('Deploy\pca_model.joblib')
        input_vector_PCA = pca.transform(input_vector)

        #Using model
        model_path = 'Deploy\KNN_best_model.pkl'
        loaded_model = load_model_from_joblib(model_path)

        # Now you can use the loaded_model for predictions on the input_vector
        if loaded_model is not None:
            predictions = loaded_model.predict(input_vector_PCA)
            box_office_list.append(float(predictions[0]))
            #print("Model Predictions:")
            #print(predictions)
    
    return box_office_list







