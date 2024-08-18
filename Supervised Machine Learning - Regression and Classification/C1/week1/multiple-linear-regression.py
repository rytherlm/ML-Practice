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
    for _ in range(iterations):
        wd, bd = derivative(xd, yd, w, b)
        w = w - (alpha * wd)
        b = b - (alpha * bd)
    return w, b




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
    iterations = 10000
    w, b = gradient(X, y, w, b, alpha, iterations)

    total_cost = cost(X, y, w, b)
    print("Total cost:", total_cost)
    print("Optimized weights:", w)
    print("Optimized bias:", b)

    example_x = np.array([2500, 3, 2])
    predicted_y = expected_value(example_x, w, b)
    print(f"Predicted house price for {example_x[0]} square feet, {example_x[1]} bedrooms, and {example_x[2]} bathrooms: ${predicted_y:,.2f}")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, label="House Prices", color="blue", marker="o")

    x1_surf, x2_surf = np.meshgrid(np.linspace(500, 4000, 10), np.linspace(1, 6, 10))
    x3_surf = np.full_like(x1_surf, 2)
    y_surf = expected_value(np.c_[x1_surf.ravel(), x2_surf.ravel(), x3_surf.ravel()], w, b).reshape(x1_surf.shape)

    ax.plot_surface(x1_surf, x2_surf, y_surf, color='red', alpha=0.5)
    ax.set_title('House Prices vs. Square Footage and Number of Bedrooms')
    ax.set_xlabel('Square Feet')
    ax.set_ylabel('Number of Bedrooms')
    ax.set_zlabel('House Price ($)')

    fig.colorbar(ax.scatter(x1, x2, y, label="House Prices", color="blue", marker="o"), ax=ax, label='House Price ($)')
    
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_plot_small_dataset():
    np.random.seed(1)
    x1 = np.random.randint(1, 10, 10)
    x2 = np.random.randint(1, 5, 10)
    x3 = np.random.randint(1, 3, 10)
    y = (x1 * 2) + (x2 * 3) + (x3 * 4) + 5

    X = np.column_stack((x1, x2, x3))

    w = np.zeros(X.shape[1])
    b = 0

    alpha = 0.01
    iterations = 1000
    w, b = gradient(X, y, w, b, alpha, iterations)

    total_cost = cost(X, y, w, b)
    print("Total cost for small dataset:", total_cost)
    print("Optimized weights for small dataset:", w)
    print("Optimized bias for small dataset:", b)

    example_x = np.array([5, 2, 1])
    predicted_y = expected_value(example_x, w, b)
    print(f"Predicted value for features {example_x[0]}, {example_x[1]}, and {example_x[2]}: {predicted_y:.2f}")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, label="Target Variable", color="blue", marker="o")

    x1_surf, x2_surf = np.meshgrid(np.linspace(1, 10, 10), np.linspace(1, 5, 10))
    x3_surf = np.full_like(x1_surf, 1)
    y_surf = expected_value(np.c_[x1_surf.ravel(), x2_surf.ravel(), x3_surf.ravel()], w, b).reshape(x1_surf.shape)

    ax.plot_surface(x1_surf, x2_surf, y_surf, color='red', alpha=0.5)
    ax.set_title('Target Variable vs. Feature 1 and Feature 2')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target Variable')

    fig.colorbar(ax.scatter(x1, x2, y, label="Target Variable", color="blue", marker="o"), ax=ax, label='Target Variable')
    
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Training and plotting large dataset...")
    train_and_plot()
    
    # print("\nTraining and plotting small dataset...")
    # train_and_plot_small_dataset()

if __name__ == "__main__":
    main()
