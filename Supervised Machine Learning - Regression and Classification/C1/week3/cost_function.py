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


    


def plot_decision_boundary(X, y, w, b):
    # Create a grid of values to plot the decision boundary
    x_values = [np.min(X[:, 0]), np.max(X[:, 0])]
    y_values = -(w[0] * np.array(x_values) + b) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', label='Data points')
    plt.plot(x_values, y_values, color='green', label='Decision boundary')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
   
    

def main():
    x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    w = np.zeros(2) 
    b = 0
    
    x_norm = z_score(x)
    print(x_norm)
    w, b = gradient(x_norm, y, w, b, 0.1, 1000)

    plot_decision_boundary(x_norm, y, w, b)

main()
