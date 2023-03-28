import numpy as np
import pandas as pd


# %% Define classes and functions

def one_hot(y, n_classes):
    return np.eye(n_classes)[y.reshape(-1)].T

def get_predictions(output):
    return np.argmax(output, 0).reshape(-1, 1)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size * 100

def print_prediction(output, Y):
    predictions = get_predictions(output)
    accuracy = get_accuracy(predictions, Y)
    print("Predictions: ", predictions.T)
    print("Accuracy: ", accuracy, "%")
    print()

class Network:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, xTrain, yTrain, alpha):
        # Forward pass
        output = self.predict(xTrain)

        # Calculate error here

        # Backward pass
        OHY = one_hot(yTrain, output.shape[0])
        gradient = output - OHY
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, alpha)
        
        return output


class Layer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Dense(Layer):
    def __init__(self, inputSize, outputSize, m):
        self.weights = np.random.randn(
            outputSize, inputSize) * np.sqrt(2 / inputSize)
        self.biases = np.zeros((outputSize, 1))
        self.m = m

    def forward(self, X):
        self.X = X
        return np.dot(self.weights, self.X) + self.biases

    def backward(self, dZ, alpha):
        self.dW = (1/self.m) * np.dot(dZ, self.X.T)
        self.db = (1/self.m) * np.sum(dZ, axis=1, keepdims=True)
        self.weights -= alpha * self.dW
        self.biases -= alpha * self.db
        return np.dot(self.weights.T, dZ)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, dZ, alpha):
        return np.multiply(dZ, self.activation_prime(self.input))


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    def __init__(self):
        def ReLU(x):
            return np.maximum(0, x)

        def ReLU_prime(x):
            return x > 0

        super().__init__(ReLU, ReLU_prime)


class Softmax(Layer):
    def forward(self, Z):
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            raise ValueError("Input array contains NaN or infinity values")
        
        Z_max = np.max(Z, axis=0)
        exp = np.exp(Z - Z_max)
        expsum = np.sum(exp, axis=0)
        return exp / expsum

    def backward(self, dZ, learning_rate):
        return dZ



# %% Read data

trainData = pd.read_csv("data_folder\mnist_train.csv").to_numpy().T
testData = pd.read_csv("data_folder\mnist_test.csv").to_numpy().T

nTrain, mTrain = trainData.shape
nTest, mTest = testData.shape

yTrain = trainData[0].reshape(mTrain, 1)
xTrain = trainData[1:nTrain] / 255

yTest = testData[0].reshape(mTest, 1)
xTest = testData[1:nTest] / 255


# %% Initialize network

nn = Network([
    Dense(28*28, 64, mTrain),
    ReLU(),
    Dense(64, 64, mTrain),
    ReLU(),
    Dense(64, 10, mTrain),
    Softmax()
])


# %% Train network

epochs = 101
alpha = 0.05
batch_size = 32
n_batches = int(np.ceil(mTrain / batch_size))

for i in range(epochs):
    permutation = np.random.permutation(mTrain)
    shuffled_X = xTrain[:, permutation]
    shuffled_Y = yTrain[permutation]
    for j in range(n_batches):
        
        # Get batch of training data
        start_idx = j * batch_size
        end_idx = min(start_idx + batch_size, mTrain)
        
        X = shuffled_X[:, start_idx:end_idx]
        Y = shuffled_Y[start_idx:end_idx]
        
        # Train on batch
        output = nn.train(X, Y, alpha)
        
    if (i % 5 == 0):
        print("Epoch: ", i)
        print_prediction(nn.predict(xTrain), yTrain)


# %% Test network

output = nn.predict(xTest)
print_prediction(output, yTest)
