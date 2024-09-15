import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import sigmoid

# Define sigmoidnp function
def sigmoidnp(z):
    return 1 / (1 + np.exp(-z))

# Define plt_logistic function to plot logistic regression results
def plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
               edgecolors=["red"], lw=3)
    
    x_vals = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_vals = model(x_vals).numpy()  # Directly call the model and convert to numpy array
    ax.plot(x_vals, y_vals, 'b-', label="Logistic Regression")

    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Logistic Regression Plot')
    ax.legend(fontsize=12)
    plt.show()

def main():
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
    Y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

    pos = Y_train == 1
    neg = Y_train == 0

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
               edgecolors=["red"], lw=3)

    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('One Variable Plot')
    ax.legend(fontsize=12)

    model = Sequential([
        Dense(1, input_dim=1, activation='sigmoid', name='L1')
    ])

    logistic_layer = model.get_layer('L1')
    w, b = logistic_layer.get_weights()

    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    logistic_layer.set_weights([set_w, set_b])
    
    a1 = model(X_train[0].reshape(1, 1)).numpy()  # Directly call the model and convert to numpy
    print(a1)
    
    alog = sigmoidnp(np.dot(set_w, X_train[0].reshape(1, 1)) + set_b)
    print(alog)
    
    plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)

main()
