import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from model_utils import *

### put the activation function here because it doesn't work when you put it in an other file(to clean later) ##"

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def RELU(Z):
    return max(0, Z)

def Tanh(Z):
    return 2 * (1 / (1 + np.exp(-Z))) - 1



class neurone():
    def __init__(self, X, y, act_fct):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
        #self.act_fct = act_fct

    def update(self, dW, db, learning_rate):
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db


class perceptron(neurone):

    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.neuron = neurone(X, y)
        loss = []

        for i in range(self.n_iter):
            A = ModelUtils.model(X, self.neuron.W, self.neuron.b, sigmoid)
            loss.append(ModelUtils.log_loss(A, y))
            dW, db = ModelUtils.gradients(A, X, y)
            self.neuron.update(dW, db, self.learning_rate)

        self.loss = loss


    def plot_loss(self):
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(self.loss)

    def predict(self, X, threshold = 0.5):
        prediction = ModelUtils.model(X, self.neuron.W, self.neuron.b, sigmoid)
        return 1 * (prediction >= threshold)

    def predict_proba(self,X):
        prediction = ModelUtils.model(X, self.neuron.W, self.neuron.b, sigmoid)
        return prediction


################# application image ############

import h5py
import numpy as np

"""

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:])  # your train set features
    y_train = np.array(train_dataset["Y_train"][:])  # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:])  # your train set features
    y_test = np.array(test_dataset["Y_test"][:])  # your train set labels

    return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = load_data()


plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()


########## reshape and normalization #####

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = y_train.astype("int64")
y_test = y_test.astype("int64")

X_train_scale = X_train.copy().astype('float64')
X_test_scale = X_test.copy().astype('float64')


def MinMaxScaler(X):
    X_std = (X - min(X)) / (max(X) - min(X))
    return X_std

for i in range(X_train.shape[0]):
    X_train_scale[i] = MinMaxScaler(X_train[i])

for i in range(X_test.shape[0]):
    X_test_scale[i] = MinMaxScaler(X_test[i])

neurone2 = perceptron(learning_rate = 0.01, n_iter =1000)

neurone2.fit(X_train_scale, y_train)
neurone2.plot_loss()
pred = neurone2.predict(X_test_scale)

print("l'accuracy du neurone est de :",accuracy_score(y_test, pred))


"""