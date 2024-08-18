import matplotlib.pyplot as plt
import numpy as np

def expected_value(X, w, b):
    return np.dot(X, w) + b

def derivative(xd, yd, w, b):
    m, n = xd.shape
    wd, bd = np.zeros(n), 0
    for i in range(m):
        err = (expected_value(xd[i], w, b) - yd[i])
        for j in range(n):
            wd[j] = wd[j] + err * xd[i, j]
        bd += err
    return (wd * (1 / m)), (bd * (1 / m))

def gradient(xd, yd, w, b, alpha, iterations):
    costs = []
    for i in range(iterations):
        wd, bd = derivative(xd, yd, w, b)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
        costs.append(cost(xd, yd, w, b))
    return w, b, costs

def cost(X, y, w, b):
    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        fx = (np.dot(X[i], w)) + b
        total_cost += (fx - y[i])**2   
    return (1 / (2 * m)) * total_cost



def train_and_plot():
    np.random.seed(0)
    x1 = np.random.randint(500, 4000, 100)
    x2 = np.random.randint(1, 6, 100)
    x3 = np.random.randint(1, 4, 100)
    noise = np.random.normal(0, 50000, 100)
    y = (x1 * 200) + (x2 * 30000) + (x3 * 20000) + 50000 + noise

    X = np.column_stack((x1, x2, x3))

    w = np.zeros(X.shape[1])
    b = 0

    alpha = 1e-10
    iterations = 5000
    w, b, costs = gradient(X, y, w, b, alpha, iterations)

    total_cost = cost(X, y, w, b)
    print("Total cost:", total_cost)
    print("Optimized weights:", w)
    print("Optimized bias:", b)

    example_x = np.array([2500, 3, 2])
    predicted_y = expected_value(example_x, w, b)
    print(f"Predicted house price for {example_x[0]} square feet, {example_x[1]} bedrooms, and {example_x[2]} bathrooms: ${predicted_y:,.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), costs, label='Cost over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Training and plotting dataset with large number of iterations...")
    train_and_plot()

if __name__ == "__main__":
    main()
