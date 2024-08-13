import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Generate prime numbers up to 100,000
limit = 10000000 
primes = generate_primes(limit)

# Create dataset where X is a prime number and y is the next prime number
X = np.array(primes[:-1]).reshape(-1, 1)  # Input features
y = np.array(primes[1:])  # Labels (next prime number)

# Train the model using scikit-learn's LinearRegression
model = LinearRegression()
model.fit(X, y)

# Predict the y values using the trained model
y_pred = model.predict(X)

# Plotting the data points and the model's prediction line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Prime Numbers')
plt.plot(X, y_pred, color='red', label='Model Prediction Line')
plt.xlabel('Prime Number')
plt.ylabel('Next Prime Number')
plt.title('Prime Number Prediction with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Output the coefficients and intercept
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
