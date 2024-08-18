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

def derivative(X, yd, w, b):
    m, n = X.shape
    wd, bd = np.zeros(n), 0
    for i in range(m):
        err = sigmoid(expected_value(X[i], w, b)) - yd[i]
        for j in range(n):
            wd[j] += err * X[i, j]
        bd += err
    return (wd * (1 / m)), (bd * (1 / m))

def gradient(X, yd, w, b, alpha, iterations):
    for _ in range(iterations):
        wd, bd = derivative(X, yd, w, b)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
    return w, b

def cost(X, y, w, b):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        fx = sigmoid(np.dot(X[i], w) + b)
        total_cost += y[i] * np.log(fx) + (1 - y[i]) * np.log(1 - fx)
    return (-1 / m) * total_cost

def data(x, w, b):
    return sigmoid(expected_value(x, w, b))

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
    w, b = gradient(X_norm, y, np.array([0, 0]), 0, 0.1, 10000)
    plot_decision_boundary(X_norm, y, w, b)

main()