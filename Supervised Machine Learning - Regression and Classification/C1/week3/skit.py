import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression





def main():
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    
    model = LogisticRegression()
    model.fit(X, y)
    y_predict = model.predict(X)
    print(y_predict)
    print(model.score(X, y))
    plt.scatter(X[:, 0], X[:, 1],c=y,cmap='Wistia')
    plt.plot(X, y_predict, color='red')
    plt.show()
        
main()