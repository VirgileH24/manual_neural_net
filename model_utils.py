import numpy as np
import matplotlib.pyplot as plt


class model_utils():

    # Activation Fonction
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def RELU(Z):
        return max(0, Z)

    @staticmethod
    def Tanh(Z):
        return 2 * (1 / (1 + np.exp(-Z))) - 1


    # metrics
    @staticmethod
    def log_loss(A, y):
        epsilon = 1e-15
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    # model
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










