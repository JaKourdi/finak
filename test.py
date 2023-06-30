
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Initialize weights and biases for hidden layers
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.weights.append(np.random.randn(input_size, hidden_sizes[i]))
            else:
                self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x.clip(max=100)))

    @staticmethod
    def _softmax(x):
        max_val = np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(x - max_val)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def _one_hot_encode(y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self._sigmoid(z)
            activations.append(a)
        return activations

    def predict(self, x):
        activations = self.forward(x)
        output = self._softmax(np.dot(activations[-1], self.weights[-1]) + self.biases[-1])
        return np.argmax(output, axis=1)

    def backward(self, x, y, learning_rate):
        activations = self.forward(x)
        delta = activations[-1] - self._one_hot_encode(y, self.output_size)

        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._sigmoid_derivative(activations[i])

    def fit(self, x, y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            self.backward(x, y, learning_rate)

    def score(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == y)

    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)


def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the input data
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    # Convert labels to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Define the neural network
    input_size = x_train.shape[1]
    output_size = np.max(y_train) + 1
    hidden_sizes = [128, 64]  # Adjust hidden layer sizes as needed

    # Create and train the neural network
    model = NeuralNetwork(input_size, hidden_sizes, output_size)
    model.fit(x_train, y_train, epochs=20, learning_rate=0.01)  # Adjust epochs and learning rate as needed

    # Evaluate the model
    accuracy = model.score(x_test, y_test)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
