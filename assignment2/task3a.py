import numpy as np
import utils
import typing
np.random.seed(1)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def pre_process_images(X: np.ndarray, mu, sigma):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        mu: mean of whole training set
        sigma: standard deviation of whole training set
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # normalize input
    X = (X-mu)/sigma

    # apply bias trick
    X_with_bias = np.zeros((X.shape[0], X.shape[1] + 1))
    X_with_bias[:, :-1] = X
    X_with_bias[:, -1] = 1  # can we just set it to 1 at the beginning?

    X = X_with_bias
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    L = - np.mean(np.sum(targets* np.log(outputs), axis = 1))

    return L


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
        i = 1
        for w in self.ws:
            z = y.dot(w)
            zs.append(z)
            if i == 1:
                y = sigmoid(z)
            else:
                y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
            ys.append(y)
            i += 1

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
        dC_dw2 = np.dot(np.transpose(ys[1]), delta_k) / len(X)

        # Hidden layer backpropagation
        delta_j = sigmoid_prime(zs[0]) * np.dot(delta_k, np.transpose(self.ws[1]))
        dC_dw1 = np.dot(np.transpose(X), delta_j) / len(X)

        self.grads.append(dC_dw1)
        self.grads.append(dC_dw2)

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    Y_encoded = np.zeros((Y.shape[0], num_classes))
    Y_encoded[range(len(Y)), Y.squeeze()] = 1

    return Y_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


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

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)

# # Output layer backpropagation for random sample
# random_sample = np.random.randint(0, len(X))
# delta_k = outputs - targets
# dC_dw2 = np.dot((ys[1][random_sample, :]).reshape(64, 1),
#                 delta_k[random_sample, :].reshape(1, 10))  # test klappt nur mit dem / len(X) aber stimmt das?
#
# # Hidden layer backpropagation for random sample
# delta_j = sigmoid_prime(zs[0]) * np.dot(delta_k, np.transpose(self.ws[1]))
# dC_dw1 = np.dot(X[random_sample, :].reshape(785, 1),
#                 delta_j[random_sample, :].reshape(1, 64))  # test klappt nur mit dem / len(X) aber stimmt das
