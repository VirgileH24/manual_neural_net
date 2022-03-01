import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from model_utils import *




class perceptron():

    def __init__(self, learning_rate, n_iter, n_neurone_1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_neurone_1 = n_neurone_1

    def __initialisation(self, n0, n1,n2):
        W1 = np.random.randn(n1, n0)
        b1 = np.random.randn(n1, 1)
        W2 = np.random.randn(n2, n1)
        b2 = np.random.randn(n2, 1)

        parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

        return parameters

    def __forward_propagation(self, X, parameters):
        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

        print("W1Shape",W1.shape,"Xshape",X.shape,"b1shape",b1.shape)
        Z1 = W1.dot(X) + b1
        A1 = 1 / (1 + np.exp(-Z1))
        print("W2shape",W2.shape, "A1shape",A1.shape)
        Z2 = W2.dot(A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))

        activations = {"A1": A1, "A2": A2}
        return activations

    def __back_propagation(self, X, y, activations,parameters):
        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
        A1, A2 = activations["A1"], activations["A2"]

        # 2 eme couche
        m = y.shape[1]
        dZ2 = A2 - y
        dW2 = 1 / m * dZ2.dot(A1.T)
        print("dz2shape", dZ2.shape, "dw2Shape", dW2.shape)
        db2 = 1 / m * np.sum(dZ2 , axis = 1, keepdims = True)



        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        print("dz1shape", dZ1.shape)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return gradients

    def __update(self, parameters, gradients, learning_rate):
        print(parameters["W1"])
        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
        dW1, db1, dW2, db2 = gradients["dW1"], gradients["db1"], gradients["dW2"], gradients["db2"]

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * 1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def fit(self, X_train, y_train):
        n0 = X_train.shape[0]
        n1 = self.n_neurone_1
        n2 = y_train.shape[0]
        self.parameters = self.__initialisation(n0, n1, n2)
        loss = []

        for i in range(self.n_iter):

            activations = self.__forward_propagation(X_train, self.parameters)
            gradients = self.__back_propagation(X_train, y_train, activations, self.parameters)
            loss.append(ModelUtils.log_loss(activations["A2"], y_train))
            self.parameters = self.__update(self.parameters, gradients, self.learning_rate)

        self.loss = loss


    def plot_loss(self):
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(self.loss)

    def predict(self, X, threshold = 0.5):
        activations = self.__forward_propagation(X, self.parameters)
        prediction = activations["A2"]
        return 1 * (prediction >= threshold)

    def predict_proba(self, X):
        activations = self.__forward_propagation(X, self.parameters)
        proba = activations["A2"]
        return proba

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