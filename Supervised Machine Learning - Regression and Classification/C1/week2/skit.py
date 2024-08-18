import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_house_data(file_path='data/houses.txt'):
    with open(file_path, 'r') as file:
        data_lines = file.readlines()

    # Convert the data into a 2D list of floats
    data_values = [list(map(float, line.split(','))) for line in data_lines]
    
    # Convert the list to a NumPy array
    data_array = np.array(data_values)
    
    # Separate the features (X) and the target (y)
    X_train = data_array[:, :-1]  # All columns except the last one
    y_train = data_array[:, -1]   # The last column
    
    # Return the features and target
    return X_train, y_train

# def main():
#     # Load the data
#     X_train, y_train = load_house_data()

#     # Initialize the scaler and scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
    
#     # Initialize and train the model
#     model = SGDRegressor(max_iter=10000)
#     model.fit(X_train_scaled, y_train)
    
#     # Predict the values using the trained model
#     y_pred = model.predict(X_train_scaled)
    
#     # Plot the actual vs predicted values
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_train, y_pred, color='blue', label='Predicted vs Actual')
    # dd
#     # Plot a diagonal line for reference
#     plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfect Prediction')
    
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.title('Actual vs Predicted House Prices')
#     plt.legend()
#     plt.show()
    
def main():
    #here
    # Load the data
    X_train, y_train = load_house_data()

    # Initialize and train the Linear Regression model
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    
    # Predict the values using the trained model
    y_predict = linear.predict(X_train)
    
    # Plot the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_predict, color='blue', label='Predicted vs Actual')
    
    # Plot a diagonal line for reference
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.show()
    
    # Print an example of a prediction for the first sample
    print(f"Actual value: {y_train[0]:.2f}, Predicted value: {y_predict[0]:.2f}")

main()
