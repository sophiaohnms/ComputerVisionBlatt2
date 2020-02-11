import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    # Task 2c
    output = model.forward(X)
    rounded_output = np.around(output)
    difference = targets - rounded_output
    num_correct = len(np.where(difference == 0)[0])
    accuracy = num_correct/len(output)

    return accuracy

def early_stopping(num_increases: int, is_val_loss_increasing: list, val_loss: dict, global_step: int, num_steps_per_val: int):
    """
    Functions that measures if the validation loss increased consistently after
    passing through 20% of the train dataset num_increases times.
    :param num_increases: number of increases of validation loss that should stopp the loop
    :param is_val_loss_increasing: list of length num_increases that contains only Bools, at beginning False
    :param val_loss: dictionary containing the validation loss
    :param global_step: step in the outer loop
    :return: True if the outer loop should break, otherwise false
    """

    value_list = []
    for step in np.arange(global_step - num_steps_per_val * num_increases, global_step, num_steps_per_val):
        value_list.append(val_loss[step])
    sorted_list = sorted(value_list)

    if sorted_list == value_list:
        print(value_list)
        return True
    else:
        return False


    # if val_loss[global_step] > val_loss[global_step - num_steps_per_val]:
    #     for i in range(len(is_val_loss_increasing)):
    #         # go through is_val_loss_increasing list and set next non True
    #         # element to True
    #         if is_val_loss_increasing[i] is False:
    #             is_val_loss_increasing[i] = True
    #             break
    #     # if there was an increase for 5 times in a row return True to break the outer loop
    #     if is_val_loss_increasing == [True]*num_increases:
    #         for step in np.arange(global_step-num_steps_per_val*num_increases, global_step+1, num_steps_per_val):
    #             print(val_loss[step])
    #         return True
    #     else:
    #         return False
    # # if the increase is not consistent reset the is_val_loss_increasing list
    # else:
    #     is_val_loss_increasing = [False]*num_increases


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda, X_train.shape[0])

    # initialize weights and outputs
    model.w = np.zeros((785,1))

    # for early stopping
    is_val_loss_increasing = [False] * num_increases

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # forward and backward pass
            output = model.forward(X_batch)
            model.backward(X_batch, output, Y_batch)

            # update weights
            model.w = model.w - learning_rate * model.grad

            # Track training loss continuously
            output_train = model.forward(X_train)
            _train_loss = cross_entropy_loss(Y_train, output_train)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                output_val = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, output_val)
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

                # early stopping
                stopping = False

                if with_stopping == True and global_step > num_increases * num_steps_per_val:
                    stopping = early_stopping(num_increases, is_val_loss_increasing, val_loss, global_step, num_steps_per_val)

                if with_stopping == True and stopping is True:
                    break

            global_step += 1

        if with_stopping == True and stopping is True:
            print('Epoch =', epoch)
            break

    return model, train_loss, val_loss, train_accuracy, val_accuracy

with_stopping = False
num_increases = 5

# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

# Preprocessing
X_train = pre_process_images(X_train)
X_test = pre_process_images(X_test)
X_val = pre_process_images(X_val)

# hyperparameters
num_epochs = 500
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Test Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))


print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))

# Plot loss

utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.ylim([0, .4])
plt.xlim(0,2000)
plt.legend()
plt.savefig("binary_train_loss_stopping=%s_epochs=%s_zoom.png" % (with_stopping, str(num_epochs)))
plt.show()


# Plot accuracy
plt.ylim([0.93, .99])
plt.xlim(0,2000)
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")

plt.legend()
plt.savefig("binary_train_accuracy_stopping=%s_epochs=%s_zoom.png" % (with_stopping, str(num_epochs)))
plt.show()

# 3 b)
# for l2_reg_lambda in [1.0, 0.1, 0.01, 0.001]:
#     model, train_loss, val_loss, train_accuracy, val_accuracy = train(
#         num_epochs=num_epochs,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         l2_reg_lambda=l2_reg_lambda)
#     utils.plot_loss(val_accuracy, r'$\lambda$ = %s' % str(l2_reg_lambda))
# plt.legend()
# plt.savefig("binary_validation_accuracy_regression.png")
# plt.show()

# 3 c)
# for l2_reg_lambda in [1.0, 0.1, 0.01, 0.001]:
#     model, train_loss, val_loss, train_accuracy, val_accuracy = train(
#         num_epochs=num_epochs,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         l2_reg_lambda=l2_reg_lambda)
#     len_vector = np.linalg.norm(model.w)
#     plt.scatter(l2_reg_lambda, len_vector)
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Length of weight vector')
# plt.grid()
# plt.savefig("L2norm.png")
# plt.show()

# 3 d)
# for l2_reg_lambda in [1.0, 0.1, 0.01, 0.001]:
#     model, train_loss, val_loss, train_accuracy, val_accuracy = train(
#         num_epochs=num_epochs,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         l2_reg_lambda=l2_reg_lambda)
#     weights_matrix = np.reshape(model.w[:-1],(28,28))
#     plt.imshow(weights_matrix)
#     plt.savefig("weights_picture_lambda=%s.png" % str(l2_reg_lambda))
#     plt.show()