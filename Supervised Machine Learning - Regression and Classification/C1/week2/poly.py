import numpy as np
import matplotlib.pyplot as plt

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

def gradient(X, yd, w, b, alpha, iterations):
    costs = []
    for _ in range(iterations):
        wd, bd = derivative(X, yd, w, b)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
        costs.append(cost(X, yd, w, b))
    return w, b, costs

def cost(X, y, w, b):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        fx = (np.dot(X[i], w)) + b
        total_cost += (fx - y[i])**2   
    return (1 / (2 *    m)) * total_cost


def main():
    x = np.arange(0, 20, 1)
    y = np.cos(x/2)
    print(y)
    X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
    X_norm = z_score(X)
    # x_features = ["x", "x^2", "x^3"]
    # print(X)

    # fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    # for i in range(len(ax)):
    #     ax[i].scatter(X[:, i], y)
    #     ax[i].set_xlabel(x_features[i])

    # plt.show()
    
    w = np.zeros(X_norm.shape[1])
    b = 0
    iterations = 10000
    
    
    w,b,costs = gradient(X_norm,y,w,b,1e-1,iterations)
    
    
    plt.figure(figsize=(10, 6))
    # plt.plot(range(iterations), costs, label='Cost over iterations')
    plt.scatter(x,y)
    plt.plot(x,X_norm@w+b)
    plt.show()
    
    
    
    
    
    

main()