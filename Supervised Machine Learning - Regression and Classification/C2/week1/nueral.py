import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential

def main():
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix

    pos = Y_train == 1
    neg = Y_train == 0

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
               edgecolors=["red"], lw=3)

    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('one variable plot')
    ax.legend(fontsize=12)
    # plt.show()

    model = Sequential(
        [
            Dense(1, input_dim=1, activation='sigmoid', name='L1')
        ]
    )

    model.summary()

main()
