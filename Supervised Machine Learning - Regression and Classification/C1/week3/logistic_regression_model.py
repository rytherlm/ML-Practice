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
        fx = (np.dot(X[i], w)) + b
        total_cost += (fx - y[i])**2   
    return (1 / (2 *    m)) * total_cost



def data(x, w, b):
    ls = []
    for i in x:
        ls.append(sigmoid(expected_value(i, w, b)))
    return np.array(ls)

def main():
    x_train = np.array([0., 1, 2, 3, 4, 5]).reshape(-1, 1)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    w = np.zeros((1))
    b = 0
    
    x_norm = z_score(x_train)
    w, b = gradient(x_norm, y_train, w, b, 0.1, 1000)
    X = data(x_norm, w, b)
    
    plt.scatter(x_train, y_train, marker='o', color='blue', label='Original data')
    plt.plot(x_train, X, color='red', label='Model prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

main()
