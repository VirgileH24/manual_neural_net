import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def RELU(Z):
    return max(0,Z)

def Tanh(Z):
    return 2 * sigmoid(Z) - 1


class model_utils():

    @staticmethod
    def log_loss(A, y):
        epsilon = 1e-15
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    @staticmethod
    def initialisation(X):
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return (W, b)

    @staticmethod
    def model(X, W, b, act_fct):
        Z = X.dot(W) + b
        A = act_fct(Z)
        return A

    @staticmethod
    def gradients(A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)

    @staticmethod
    def update(dW, db, W, b, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        print(W)
        return (W, b)


class neurone():
    def __init__(self, X, y, act_fct):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
        self.act_fct = act_fct

    def update(self, dW, db, learning_rate):
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db


class perceptron(neurone):

    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.neuron = neurone(X, y, sigmoid)
        loss = []

        for i in range(self.n_iter):
            A = model_utils.model(X, self.neuron.W, self.neuron.b,sigmoid)
            loss.append(model_utils.log_loss(A, y))
            dW, db =model_utils.gradients(A, X, y)
            self.neuron.update(dW, db, self.learning_rate)

        self.loss = loss


    def plot_loss(self):
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(self.loss)

    def predict(self, X, threshold = 0.5):
        prediction = model_utils.model(X, self.neuron.W, self.neuron.b, sigmoid)
        return 1 * (prediction >= threshold)

    def predict_proba(self,X):
        prediction = model_utils.model(X, self.neuron.W, self.neuron.b, sigmoid)
        return prediction


################### Application 1  ############

X, y  = make_blobs(n_samples=100,
                   centers=2,
                   n_features=2,
                   random_state=0)
y = y.reshape((y.shape[0], 1))

W, b = model_utils.initialisation(X)
print(model_utils.model(X,W,b,sigmoid))

perc = perceptron(
                learning_rate = 0.1,
                n_iter = 100
                )

perc.fit(X, y)

new_plant = np.array([2,1])
plt.scatter(X[:,0], X[:,1], c = y,cmap = 'summer')
plt.scatter(new_plant[0], new_plant[1], c = 'r')
#plt.show()

y_pred = perc.predict(new_plant)
print(y_pred)

perc.plot_loss()



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