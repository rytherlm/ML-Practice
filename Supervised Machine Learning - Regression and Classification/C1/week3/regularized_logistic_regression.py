import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def expected_value(X, w, b):
    return np.dot(X, w) + b

def z_score(X):
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - m) / std

def derivative(X, yd, w, b, lambda_=1):
    m, n = X.shape
    wd, bd = np.zeros(n), 0
    for i in range(m):
        fx = sigmoid(expected_value(X[i], w, b))
        err = fx - yd[i]  # Logistic regression error
        for j in range(n):
            wd[j] += err * X[i, j]
        bd += err
    
    wd = (wd / m) + ((lambda_ / m) * w)  # Regularization term for weights
    bd = bd / m
    return wd, bd

def gradient(X, yd, w, b, alpha, iterations, lambda_=1):
    costs = []
    for _ in range(iterations):
        wd, bd = derivative(X, yd, w, b, lambda_)
        w -= (alpha * wd)
        b -= (alpha * bd)
        costs.append(cost(X, yd, w, b, lambda_))
    return w, b, costs

def cost(X, y, w, b, lambda_=1):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        fx = sigmoid(expected_value(X[i], w, b))
        total_cost += y[i] * np.log(fx) + (1 - y[i]) * np.log(1 - fx)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)  # Regularization term should include 1/2m factor
    return (-1 / m) * total_cost + reg_cost

def plot_decision_boundary(X, y, w, b):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', label='Data points')
    
    # Create a meshgrid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    
    # Calculate the decision boundary
    Z = sigmoid(np.dot(np.c_[xx1.ravel(), xx2.ravel()], w) + b)
    Z = Z.reshape(xx1.shape)
    
    # Plot the decision boundary
    plt.contour(xx1, xx2, Z, [0.5], colors='g', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    X_norm = z_score(x)
    w, b,cost = gradient(X_norm, y, np.array([0.0, 0.0]), 0.0, 0.1, 10000)
    plot_decision_boundary(X_norm, y, w, b)

main()
