import numpy as np
import matplotlib.pyplot as plt

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
        err = (expected_value(X[i], w, b) - yd[i])
        for j in range(n):
            wd[j] = wd[j] + err * X[i, j]
        bd += err
    
    wd = (wd * (1 / m)) + ((lambda_ / m) * w)  # Regularization term for weights
    bd = bd * (1 / m)
    return wd, bd

def gradient(X, yd, w, b, alpha, iterations, lambda_=1):
    costs = []
    for _ in range(iterations):
        wd, bd = derivative(X, yd, w, b, lambda_)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
        costs.append(cost(X, yd, w, b, lambda_))
    return w, b, costs

def cost(X, y, w, b, lambda_=1):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        fx = (np.dot(X[i], w)) + b
        total_cost += (fx - y[i])**2  
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)  # Regularization term should include 1/2m factor
    return ((1 / (2 * m)) * total_cost) + reg_cost


def main():
    x = np.arange(0, 20, 1)
    y = np.cos(x/2)
    print(y)
    X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
    X_norm = z_score(X)    
    w = np.zeros(X_norm.shape[1])
    b = 0
    iterations = 100000
    
    
    w,b,costs = gradient(X_norm,y,w,b,1e-1,iterations,lambda_=.1)
    
    
    plt.figure(figsize=(10, 6))
    # plt.plot(range(iterations), costs, label='Cost over iterations')
    plt.scatter(x,y)
    plt.plot(x,X_norm@w+b)
    plt.show()
    
    
    
    
    
    

main()