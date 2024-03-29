from scipy.special import expit
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    return expit(x)


class NeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_neurons=50, n_categories=10, epochs=10, batch_size=100, eta=0.1,
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
        loss = -np.mean(np.log(probabilities[np.arange(X.shape[0]), y.reshape(-1, 1).astype(int)]))
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
                chosen_datapoints = data_indices[j * self.batch_size:(j + 1) * self.batch_size]
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.feed_forward()
                self.backpropagation(epoch * self.iterations + j + 1)

            # Check loss on validation data
            val_loss = self.calculate_loss(self.X_data_val, self.Y_data_val)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_weights = {
                    'hidden_weights': self.hidden_weights.copy(),
                    'hidden_bias': self.hidden_bias.copy(),
                    'output_weights': self.output_weights.copy(),
                    'output_bias': self.output_bias.copy()
                }
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print("Training stopped due to early stopping.")
                break

    def set_validation_data(self, X_val, y_val):
        self.X_data_val = X_val
        self.Y_data_val = y_val


def generate_adversarial_example(model, X, y, epsilon=0.1):
    X_adv = X.copy()
    gradients = model.calculate_gradients(X_adv, y)
    X_adv -= epsilon * np.sign(gradients)
    return X_adv


def main():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess data
    X_train = X_train.reshape((-1, 28 * 28)) / 255.0
    X_test = X_test.reshape((-1, 28 * 28)) / 255.0

    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Create and train the neural network
    nn = NeuralNetwork(X_train, y_train_onehot, n_hidden_neurons=50, n_categories=10, epochs=10, batch_size=100,
                       eta=0.1, lmbd=0.0, patience=5)
    nn.set_validation_data(X_test, y_test_onehot)
    nn.train()

    train_accuracy = nn.score(X_train, y_train)
    test_accuracy = nn.score(X_test, y_test)

    print("Part 1:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate adversarial example
    example_index = 0  # Choose an example to generate the adversarial example
    X_example = X_test[example_index]
    y_example = y_test[example_index]
    X_adversarial = generate_adversarial_example(nn, X_example, y_example, epsilon=0.1)

    # Predict the original and adversarial examples
    y_pred_original = nn.predict(X_example.reshape((1, -1)))
    y_pred_adversarial = nn.predict(X_adversarial.reshape((1, -1)))

    print("Original Image - Predicted Label:", y_pred_original)
    print("Adversarial Image - Predicted Label:", y_pred_adversarial)


if __name__ == '__main__':
    main()
