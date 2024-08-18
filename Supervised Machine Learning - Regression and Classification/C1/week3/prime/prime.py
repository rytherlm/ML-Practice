import matplotlib.pyplot as plt
import numpy as np

# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True

# Generate prime numbers up to a limit
def generate_primes(limit):
    primes = []
    for i in range(2, limit):
        if is_prime(i):
            primes.append(i)
    return primes

# Original functions for model training
def expected_value(X, w, b):
    return np.dot(X, w) + b

def derivative(X, yd, w, b):
    m, n = X.shape
    wd, bd = np.zeros(n), 0
    for i in range(m):
        err = (expected_value(X[i], w, b) - yd[i])
        for j in range(n):
            wd[j] = wd[j] + err * X[i, j]
        bd += err
    return (wd * (1 / m)), (bd * (1 / m))

def gradient(X, yd, w, b, alpha, iterations):
    for _ in range(iterations):
        wd, bd = derivative(X, yd, w, b)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
    return w, b

def z_score(X):
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - m) / std

def main():
    # Generate prime numbers up to 200,000
    limit = 200000
    primes = generate_primes(limit)
    
    # Create dataset where X is a prime number and y is the next prime number
    X = np.array(primes[:-1]).reshape(-1, 1)  # Input features
    y = np.array(primes[1:])  # Labels (next prime number)

    # Normalize the data
    X_normalized = z_score(X)

    # Initialize weights and bias
    w = np.zeros(X_normalized.shape[1])
    b = 0

   
    alpha = 0.01  # Learning rate
    iterations = 300

    # Train the model using gradient descent
    w, b = gradient(X_normalized, y, w, b, alpha, iterations)

    # Predict the y values using the trained model
    y_pred = expected_value(X_normalized, w, b)

    # Plotting the data points and the model's prediction line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Prime Numbers')
    plt.plot(X, y_pred, color='red', label='Model Prediction Line')
    plt.xlabel('Prime Number')
    plt.ylabel('Next Prime Number')
    plt.title('Prime Number Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
