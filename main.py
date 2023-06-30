import numpy as np
from tensorflow.keras.datasets import mnist
from neural_network import NeuralNetwork


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
    hidden_size = 64
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    model.fit(x_train, y_train, epochs=10, learning_rate=0.1)

    # Evaluate the model
    accuracy = model.score(x_test, y_test)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
