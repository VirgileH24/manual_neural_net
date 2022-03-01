import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from main import perceptron
from model_utils import *




### put the activation function here because it doesn't work when you put it in an other file(to clean later) ##"

## application 1

X, y  = make_blobs(n_samples=100,
                   centers=2,
                   n_features=2,
                   random_state=0)
y = y.reshape((y.shape[0], 1))



perc = perceptron(
                learning_rate = 0.1,
                n_iter = 100,
                n_neurone_1= 8
                )

perc.fit(X.T, y.T)

new_plant = np.array([2,1]).T
plt.scatter(X[:,0], X[:,1], c = y,cmap = 'summer')
plt.scatter(new_plant[0], new_plant[1], c = 'r')
plt.show()

#y_pred = perc.predict(new_plant)
#print(y_pred)

perc.plot_loss()

