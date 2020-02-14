import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    output = model.forward(X)
    result_output = np.argmax(output, axis=1)
    target = np.argmax(targets, axis=1)
    correct = target == result_output
    accuracy = np.mean(correct)

    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    global_step = 0
    for epoch in range(num_epochs):

        # Task 3 a: shuffeling
        if use_shuffle is True:
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # forward and backward pass
            output = model.forward(X_batch)
            model.backward(X_batch, output, Y_batch)

            # update weights
            if use_momentum is True:
                model.ws[0] = model.ws[0] - learning_rate * model.grads[0] - momentum_gamma * learning_rate * model.old_grads[0]
                model.ws[1] = model.ws[1] - learning_rate * model.grads[1] - momentum_gamma * learning_rate * model.old_grads[1]
            else:
                model.ws[0] = model.ws[0] - learning_rate * model.grads[0]
                model.ws[1] = model.ws[1] - learning_rate * model.grads[1]

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                output_val = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, output_val)
                val_loss[global_step] = _val_loss

                output_train = model.forward(X_train)
                _train_loss = cross_entropy_loss(Y_train, output_train)
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    bool_vector = np.array([False, False, False, False, False])
    names = ["", "with shuffeling", "with improved sigmoid", "with improved weights", "with momentum"]
    color = ["red", "blue", "green", "orange", "purple"]

    # Plot loss
    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_ylim([0.1, 0.5])
    ax1.set_xlabel("Number of gradient steps")
    ax1.set_ylabel("Cross Entropy Loss")
    # Plot accuracy
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_ylim([0.9, 1.0])
    ax2.set_xlabel("Number of gradient steps")
    ax2.set_ylabel("Accuracy")

    for i in range(5):
        # Load dataset
        validation_percentage = 0.2
        X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
            validation_percentage)

        # pre processing
        mu = np.mean(X_train)
        sigma = np.std(X_train)
        X_train = pre_process_images(X_train, mu, sigma)
        Y_train = one_hot_encode(Y_train, 10)
        X_val = pre_process_images(X_val, mu, sigma)
        Y_val = one_hot_encode(Y_val, 10)
        X_test = pre_process_images(X_test, mu, sigma)
        Y_test = one_hot_encode(Y_test, 10)

        # Settings for task 3. Keep all to false for task 2.
        use_shuffle = bool_vector[0]
        use_improved_sigmoid = bool_vector[1]
        use_improved_weight_init = bool_vector[2]
        use_momentum = bool_vector[3]

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        model, train_loss, val_loss, train_accuracy, val_accuracy = train(
            model,
            [X_train, Y_train, X_val, Y_val, X_test, Y_test],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_shuffle=use_shuffle,
            use_momentum=use_momentum,
            momentum_gamma=momentum_gamma)

        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Test Cross Entropy Loss:",
              cross_entropy_loss(Y_test, model.forward(X_test)))

        print("Final Train accuracy:",
              calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))
        print("Final Test accuracy:",
              calculate_accuracy(X_test, Y_test, model))

        utils.plot_loss(train_loss, "Training Loss" + " " + names[i], ax = ax1, c = color[i], ls = "-")
        utils.plot_loss(val_loss, "Validation Loss" + " " + names[i], ax = ax1, c = color[i], ls = "--")

        utils.plot_loss(train_accuracy, "Training Accuracy"  + " " + names[i], ax = ax2, c = color[i], ls = "-")
        utils.plot_loss(val_accuracy, "Validation Accuracy" + " " + names[i], ax = ax2, c = color[i], ls = "--")

        print(use_shuffle, use_improved_sigmoid, use_improved_weight_init, use_momentum)
        #bool_vector[i] = True

    ax1.legend()
    ax2.legend()
    fig.savefig("softmax_train_graph_task3_alloff.png")
    fig.show()
