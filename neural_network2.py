import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_neurons=50, n_categories=10, epochs=10, batch_size=50, eta=0.1,
                 lmbd=0.0, patience=5):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.patience = patience  # Number of epochs to wait for improvement
        self.best_loss = np.inf
        self.best_weights = None
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_output_weights = np.zeros_like(self.output_weights)
        self.v_output_weights = np.zeros_like(self.output_weights)
        self.m_output_bias = np.zeros_like(self.output_bias)
        self.v_output_bias = np.zeros_like(self.output_bias)
        self.m_hidden_weights = np.zeros_like(self.hidden_weights)
        self.v_hidden_weights = np.zeros_like(self.hidden_weights)
        self.m_hidden_bias = np.zeros_like(self.hidden_bias)
        self.v_hidden_bias = np.zeros_like(self.hidden_bias)

    def feed_forward(self):
        self.z_h = np.dot(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        self.z_o = np.dot(self.a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        z_h = np.dot(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)
        z_o = np.dot(a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self, t):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.dot(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        self.output_weights_gradient = np.dot(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        self.hidden_weights_gradient = np.dot(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights
        self.m_output_weights = self.beta1 * self.m_output_weights + (1 - self.beta1) * self.output_weights_gradient
        self.v_output_weights = self.beta2 * self.v_output_weights + (1 - self.beta2) * np.square(
            self.output_weights_gradient)
        self.m_output_bias = self.beta1 * self.m_output_bias + (1 - self.beta1) * self.output_bias_gradient
        self.v_output_bias = self.beta2 * self.v_output_bias + (1 - self.beta2) * np.square(self.output_bias_gradient)
        self.m_hidden_weights = self.beta1 * self.m_hidden_weights + (1 - self.beta1) * self.hidden_weights_gradient
        self.v_hidden_weights = self.beta2 * self.v_hidden_weights + (1 - self.beta2) * np.square(
            self.hidden_weights_gradient)
        self.m_hidden_bias = self.beta1 * self.m_hidden_bias + (1 - self.beta1) * self.hidden_bias_gradient
        self.v_hidden_bias = self.beta2 * self.v_hidden_bias + (1 - self.beta2) * np.square(self.hidden_bias_gradient)
        m_output_weights_corrected = self.m_output_weights / (1 - self.beta1 ** t)
        v_output_weights_corrected = self.v_output_weights / (1 - self.beta2 ** t)
        m_output_bias_corrected = self.m_output_bias / (1 - self.beta1 ** t)
        v_output_bias_corrected = self.v_output_bias / (1 - self.beta2 ** t)
        m_hidden_weights_corrected = self.m_hidden_weights / (1 - self.beta1 ** t)
        v_hidden_weights_corrected = self.v_hidden_weights / (1 - self.beta2 ** t)
        m_hidden_bias_corrected = self.m_hidden_bias / (1 - self.beta1 ** t)
        v_hidden_bias_corrected = self.v_hidden_bias / (1 - self.beta2 ** t)
        self.output_weights -= self.eta * m_output_weights_corrected / (
                    np.sqrt(v_output_weights_corrected) + self.epsilon)
        self.output_bias -= self.eta * m_output_bias_corrected / (np.sqrt(v_output_bias_corrected) + self.epsilon)
        self.hidden_weights -= self.eta * m_hidden_weights_corrected / (
                    np.sqrt(v_hidden_weights_corrected) + self.epsilon)
        self.hidden_bias -= self.eta * m_hidden_bias_corrected / (np.sqrt(v_hidden_bias_corrected) + self.epsilon)

    def calculate_loss(self, X, y):
        probabilities = self.feed_forward_out(X)
        loss = -np.mean(np.log(probabilities[np.arange(X.shape[0]), y]))
        if self.lmbd > 0.0:
            weights_sum = np.sum(np.square(self.hidden_weights)) + np.sum(np.square(self.output_weights))
            loss += 0.5 * self.lmbd * weights_sum
        return loss

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def train(self):
        data_indices = np.arange(self.n_inputs)
        patience_count = 0  # Counter for patience
        for epoch in range(self.epochs):
            np.random.shuffle(data_indices)
            for j in range(self.iterations):
                chosen_datapoints = data_indices[j * self.batch_size: (j + 1) * self.batch_size]
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.Y_data = np.eye(self.n_categories)[self.Y_data]
                self.feed_forward()
                self.backpropagation(t=(epoch + 1))

            # Calculate loss on validation set for early stopping
            val_loss = self.calculate_loss(self.X_data_val, self.Y_data_val)

            if val_loss < self.best_loss:
                # Improvement in validation loss
                self.best_loss = val_loss
                self.best_weights = {
                    "hidden_weights": np.copy(self.hidden_weights),
                    "hidden_bias": np.copy(self.hidden_bias),
                    "output_weights": np.copy(self.output_weights),
                    "output_bias": np.copy(self.output_bias),
                }
                patience_count = 0  # Reset patience count
            else:
                # No improvement in validation loss
                patience_count += 1

            if patience_count == self.patience:
                # Early stopping condition met
                print(f"Early stopping at epoch {epoch + 1}...")
                break

        if self.best_weights is not None:
            # Restore the best weights found during training
            self.hidden_weights = self.best_weights["hidden_weights"]
            self.hidden_bias = self.best_weights["hidden_bias"]
            self.output_weights = self.best_weights["output_weights"]
            self.output_bias = self.best_weights["output_bias"]


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Splitting training data into training and validation sets
    split_index = int(0.8 * len(x_train))
    x_train_split = x_train[:split_index]
    y_train_split = y_train[:split_index]
    x_val_split = x_train[split_index:]
    y_val_split = y_train[split_index:]

    model = NeuralNetwork(x_train_split, y_train_split, n_hidden_neurons=50, n_categories=10, epochs=10, batch_size=50,
                          eta=0.1, lmbd=0.01, patience=5)
    model.X_data_val = x_val_split
    model.Y_data_val = y_val_split
    model.train()

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)

    print("Part 1:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")



if __name__ == '__main__':
    main()
