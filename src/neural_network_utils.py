import numpy as np
from matplotlib import pyplot as plt


def plot_loss_curve(loss_function_dict):
    """ plotting the loss function and test accuracy corresponding to the number of iterations """
    plt.plot(loss_function_dict.keys(), loss_function_dict.values())
    plt.ylabel('Loss function')
    plt.xlabel('Number of iterations')
    plt.xticks(rotation=60)
    plt.title('Loss function w.r.t. number of iterations')
    plt.show()


def plot_accuracy_curve(validation_dict: dict, final_test_accuracy: float = None):
    title_txt = 'Validation accuracy w.r.t. number of iterations'
    if final_test_accuracy is not None:
        title_txt += f"\nFinal test accuracy: {final_test_accuracy}"

    plt.plot(validation_dict.keys(), validation_dict.values())
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Number of iterations')
    plt.xticks(rotation=60)
    plt.title(title_txt)
    plt.show()


def l_rate_scheduler(base_rate, current_iter, num_iterations):
    return base_rate * 10 ** (-np.floor(current_iter / num_iterations * 5))


def transform_mnist_x_data(x_vals):
    # Transform Train Data
    x_train = x_vals.reshape(x_vals.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    return x_train


def apply_activation_function(x, type='ReLU'):
    # implement the activation function
    if type == 'ReLU':
        return np.array([i if i > 0 else 0 for i in np.squeeze(x)])
    elif type == 'Sigmoid':
        return 1 / (1 + np.exp(-x))
    elif type == 'tanh':
        return 1 - (np.tanh(x)) ** 2
    else:
        raise TypeError('Invalid activation function')


def softmax(x):
    # implement the softmax function
    return 1 / sum(np.exp(x)) * np.exp(x)


def cross_entropy_error(v, y):
    # implement the cross entropy error
    # x_vals = np.squeeze(x)
    return -np.log(v[0][y])


def apply_derivative_activation_function(x, type='ReLU'):
    # implement the activation function
    if type == 'ReLU':
        return np.array([1 if i > 0 else 0 for i in np.squeeze(x)])
    elif type == 'Sigmoid':
        return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    elif type == 'tanh':
        return
    else:
        raise TypeError('Invalid derivative activation function')
