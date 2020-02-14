import numpy as np
import utils
import typing
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool,  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 784
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer



        # Initialize the weights
        self.ws = []
        prev = self.I + 1
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.random.uniform(low=-1, high=1, size=w_shape )
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]



    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        y = X
        global ys
        ys = [X]  # list to store all the activations, layer by layer
        global zs
        zs = []  # list to store all the z vectors, layer by layer

        amount_layer=range(len(neurons_per_layer))

        for j in amount_layer:
            for w in self.ws:
                z = y.dot(w)
                zs.append(z)
                if j == len(neurons_per_layer)-1:
                    y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
                else:
                    y = sigmoid(z)
                ys.append(y)


            return y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []

        # Output layer backpropagation
        delta_k = outputs - targets
        print(delta_k.shape, ys[1].shape )
        dC_dw2 = np.dot(np.transpose(ys[1]), delta_k) / len(X)

        amount_layer = range(len(neurons_per_layer)-1)

        for j in amount_layer:

            # Hidden layer backpropagation
            delta_j = sigmoid_prime(zs[0]) * np.dot(delta_k, np.transpose(self.ws[1]))
            dC_dw1 = np.dot(np.transpose(X), delta_j) / len(X)
            print(zs[0].shape, self.ws[1].shape, delta_j.shape, X.shape, dC_dw1.shape, dC_dw2.shape)

            self.grads.append(dC_dw1)
            self.grads.append(dC_dw2)

            for grad, w in zip(self.grads, self.ws):
                assert grad.shape == w.shape, \
                    f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."


def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    mu = np.mean(X_train)
    sigma = np.std(X_train)
    X_train = pre_process_images(X_train, mu, sigma)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [64, 64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)

