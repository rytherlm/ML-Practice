import matplotlib.pyplot as plt
import numpy as np

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

def z_score(X):
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - m) / std

def main():
    np.random.seed(0)
    
    # Generate a simpler synthetic data set with 3 features
    x1 = np.random.randint(1, 10, 20)
    x2 = np.random.randint(1, 5, 20)
    x3 = np.random.randint(1, 3, 20)
    noise = np.random.normal(0, 2, 20)
    y = (x1 * 3) + (x2 * 2) + (x3 * 4) + noise

    X = np.column_stack((x1, x2, x3))

    # Print the original data mean and standard deviation
    print("Original data mean:", np.mean(X, axis=0))
    print("Original data std:", np.std(X, axis=0))

    # Normalize the data
    X_normalized = z_score(X)

    # Print the normalized data mean and standard deviation
    print("Normalized data mean:", np.mean(X_normalized, axis=0))
    print("Normalized data std:", np.std(X_normalized, axis=0))

    # Initialize weights and bias
    w = np.zeros(X_normalized.shape[1])
    b = 0

    # Set hyperparameters
    alpha = 0.01  # Increased learning rate
    iterations = 300

    # Train the model using gradient descent
    w, b, costs = gradient(X_normalized, y, w, b, alpha, iterations)

    # Calculate the final cost
    total_cost = cost(X_normalized, y, w, b)
    print("Total cost:", total_cost)
    print("Optimized weights:", w)
    print("Optimized bias:", b)

    # Plotting the cost function convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), costs, label='Cost over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
