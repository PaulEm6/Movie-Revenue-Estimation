import pandas as pd
import matplotlib.pyplot as plt

budget = 'Original budget'
box_office = 'Total original B.O'

def plot_predictions(dataframe, name, save_path):
    # Extract real values and predicted values for each model
    predicted_name = box_office
    predictions = f'{name} predictions'

    real_values = dataframe[predicted_name]
    predicted_values = dataframe[predictions]

    # Create a sample pyplot chart
    plt.figure(figsize=(10, 6))

    # Plot each model's predictions against real values
    #plt.scatter(real_values, model1_predictions, label=lasso, alpha=0.7)
    plt.scatter(real_values, predicted_values, label=name, alpha=0.7)

    # Add the first bisector line
    plt.plot(real_values, real_values, color='black', linestyle='--', label='First Bisector')

    # Add labels and legend
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Comparison of Predicted Values')
    plt.legend()

    # Show the plot
    #plt.show()

    plt.savefig(save_path)


# Example usage:
# Assuming you have a DataFrame called 'df' with columns: 'Real_Values', 'Model1_Predictions', 'Model2_Predictions', 'Mode
# l3_Predictions', 'Model4_Predictions'
# Replace 'df' with your actual DataFrame name

name = 'RBFNetwork'
df = pd.read_csv(f'Dataframe\model_result\{name}.csv')
plot_predictions(df, name=name, save_path=f'Results\{name}.png')


