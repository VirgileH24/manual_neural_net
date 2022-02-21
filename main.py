import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from model_utils import *




class perceptron():

    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def __initialisation(self, X):
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return (W, b)

    def __model(self, X, W, b):
        Z = X.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        return A

    def __gradients(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)

    def __update(self, dW, db, W,b, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        print(W)
        return (W, b)

    def fit(self, X, y):
        self.W, self.b = self.__initialisation(X)
        loss = []

        for i in range(self.n_iter):

            A = self.__model(X, self.W, self.b)
            loss.append(ModelUtils.log_loss(A, y))
            dW, db = self.__gradients(A, X, y)
            print("les coeff sont ", dW, db, A)
            self.__update(dW, db,self.W,self.b, self.learning_rate)



        self.loss = loss
        #print(self.loss)


    def plot_loss(self):
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(self.loss)

    def predict(self, X, threshold = 0.5):
        prediction = self.__model(X, self.W, self.b)
        return 1 * (prediction >= threshold)

    def predict_proba(self, X):
        prediction = self.__model(X, self.W, self.b)
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