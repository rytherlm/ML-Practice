import matplotlib.pyplot as plt
import numpy as np

def expected_value(x, w, b):
    return (x * w) + b

def derivative(xd, yd, w, b):
    m = xd.shape[0]
    fw, fb = 0, 0
    for i in range(m):
        fw += (((w * xd[i]) + b) - yd[i]) * xd[i]
        fb += (((w * xd[i]) + b) - yd[i])
    return fw / m, fb / m

def gradient(xd, yd, w, b, alpha, iterations):
    for _ in range(iterations):
        fw, fb = derivative(xd, yd, w, b)
        print(fw,fb)
        w -= alpha * fw
        b -= alpha * fb
    return w, b

def cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        fx = (w * x[i]) + b
        total_cost += (fx - y[i])**2   
    return (1 / (2 * m)) * total_cost

def model(x, w, b):
    xd, yd = [], []
    m = x.shape[0]
    for i in range(m):
        xd.append(x[i])
        yd.append((x[i] * w) + b)
    return [xd, yd]

def main():
    np.random.seed(0)
    x_data = np.random.randint(500, 4000, 500)  
    noise = np.random.normal(0, 50000, 500)  
    y_data = x_data * 250 + 50000 + noise  
    
    w, b = gradient(x_data, y_data, 0, 0, 0.00000001, 10000)
    
    total_cost = cost(x_data, y_data, w, b)
    print("Total cost:", total_cost)
    print("Optimized w:", w)
    print("Optimized b:", b)
    
    # Example usage of expected_value function
    example_x = 2500
    predicted_y = expected_value(example_x, w, b)
    print(f"Predicted house price for {example_x} square feet: ${predicted_y:,.2f}")
    
    x_line, y_line = model(x_data, w, b)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="House Prices", color="blue", marker="o")
    plt.plot(x_line, y_line, label=f'Linear Regression: y = {w:.2f}x + {b:.2f}', color='red')
    
    plt.title('House Prices vs. Square Footage')
    plt.xlabel('Square Feet')
    plt.ylabel('House Price ($)')
    
    plt.legend()
    plt.legend()
    plt.legend()

if __name__ == "__main__":
    main()
