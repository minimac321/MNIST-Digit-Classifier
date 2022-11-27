import copy

import numpy as np
from tqdm import tqdm

from src.neural_network_utils import (
    apply_activation_function,
    softmax,
    apply_derivative_activation_function,
    cross_entropy_error,
    l_rate_scheduler,
)


class FeedForwardNeuralNetwork:
    first_layer = {}
    second_layer = {}

    def __init__(self, inputs: int, hidden: int, outputs: int):
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

        # initialize the model parameters, including the first and second layer
        # parameters and biases
        self.first_layer["weights"] = np.random.uniform(
            low=-np.sqrt(2 / hidden), high=np.sqrt(2 / hidden), size=(hidden, inputs)
        )
        self.first_layer["bias"] = np.random.uniform(
            low=-np.sqrt(2 / hidden), high=np.sqrt(2 / hidden), size=(hidden, 1)
        )
        self.second_layer["weights"] = np.random.uniform(
            low=-np.sqrt(2 / outputs), high=np.sqrt(2 / outputs), size=(outputs, hidden)
        )
        self.second_layer["bias"] = np.random.uniform(
            low=-np.sqrt(2 / outputs), high=np.sqrt(2 / outputs), size=(outputs, 1)
        )

    def _forward_propagation(self, x: np.array) -> dict:
        assert x.shape == (1, 784), x.shape
        # implement the forward computation, calculation of prediction list and error
        hidden_layer_input = (
            np.matmul(self.first_layer["weights"], x.T).reshape((self.hid_size, 1))
            + self.first_layer["bias"]
        )

        hidden_layer_output = np.array(apply_activation_function(hidden_layer_input)).reshape(
            (self.hid_size, 1)
        )
        final_layer_input = (
            np.matmul(self.second_layer["weights"], hidden_layer_output).reshape(
                (self.output_size, 1)
            )
            + self.second_layer["bias"]
        )
        predict_list = np.squeeze(softmax(final_layer_input))

        results = {
            "hidden_layer_input": hidden_layer_input,  # Z
            "hidden_layer_output": hidden_layer_output,  # H
            "final_layer_input": final_layer_input,  # U
            "predicted_values": predict_list.reshape((1, self.output_size)),
        }
        return results

    def _back_propagation(self, x, y, f_result: dict):
        """Implement the back propagation process, compute the gradients"""
        true_label = np.array([0] * self.output_size).reshape((1, self.output_size))
        true_label[0][y] = 1
        y_hat = true_label - f_result["predicted_values"]
        final_layer_output_error = -(y_hat).reshape((self.output_size, 1))  # dU
        bias_2_error = copy.copy(final_layer_output_error)  # db_2

        final_layer_error_gradient = np.matmul(
            final_layer_output_error, f_result["hidden_layer_output"].T
        )
        hidden_layer_input_error = np.matmul(
            self.second_layer["weights"].T, final_layer_output_error
        )
        hidden_layer_output_error = hidden_layer_input_error.reshape(
            self.hid_size, 1
        ) * apply_derivative_activation_function(f_result["hidden_layer_input"]).reshape(
            self.hid_size, 1
        )

        hidden_layer_error_gradient = np.matmul(
            hidden_layer_output_error.reshape((self.hid_size, 1)), x.reshape((1, 784))
        )

        grad = {
            "final_layer_error_gradient": final_layer_error_gradient,
            "final_layer_output_error": final_layer_output_error,
            "hidden_layer_output_error": hidden_layer_output_error,
            "hidden_layer_error_gradient": hidden_layer_error_gradient,
        }
        return grad

    def _optimize_parameters(self, b_result: dict, learning_rate: float):
        # update the hyperparameters
        self.second_layer["weights"] -= learning_rate * b_result["final_layer_error_gradient"]
        self.second_layer["bias"] -= learning_rate * b_result["final_layer_output_error"]
        self.first_layer["weights"] -= learning_rate * b_result["hidden_layer_error_gradient"]
        self.first_layer["bias"] -= learning_rate * b_result["hidden_layer_output_error"]

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        x_valid: np.array,
        y_valid: np.array,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> tuple[dict, dict]:

        loss_dict = {}
        training_scores, validation_scores = [], []
        for i_epoch in tqdm(range(epochs)):
            if batch_size is None or batch_size >= len(x_train):
                sample_idx = list(range(len(x_train)))
            else:
                sample_idx = np.random.choice(
                    range(x_train.shape[0]), size=batch_size, replace=True
                )

            # Iterate over all samples for epoch
            epoch_loss = []
            for i, selected_sample_index in enumerate(sample_idx):
                selected_sample = x_train[selected_sample_index]
                sample_label = y_train[selected_sample_index]

                f_result = self._forward_propagation(selected_sample)
                error = cross_entropy_error(f_result["predicted_values"], sample_label)
                f_result["error"] = error
                epoch_loss.append(error)

                b_result = self._back_propagation(selected_sample, sample_label, f_result)
                self._optimize_parameters(
                    b_result,
                    l_rate_scheduler(
                        base_rate=learning_rate,
                        current_iter=learning_rate,
                        num_iterations=len(sample_idx),
                    ),
                )

            loss_dict[i_epoch] = np.array(epoch_loss).mean()

            if i_epoch % 5 == 0:
                training_score = self.testing(x_train[sample_idx], y_train[sample_idx])
                training_scores.append(training_score)
                validation_score = self.testing(x_valid, y_valid)
                validation_scores.append(validation_score)
                print(f"Current epoch: {i_epoch}.\n"
                      f"Training accuracy = {training_score:.3f}. "
                      f"Validation score = {validation_score:.3f}")

        training_dict = {
            "train": {i: train_score for i, train_score in enumerate(training_scores)},
            "validation": {i: val_score for i, val_score in enumerate(validation_scores)},
        }
        print("Neural Network training finished")
        return training_dict, loss_dict

    def predict(self, input_data):
        if input_data.shape == (28, 28):
            input_data = input_data.reshape((1, 784))

        f_result = self._forward_propagation(input_data)
        predicted_label = np.argmax(np.squeeze(f_result["predicted_values"]))
        return predicted_label

    def testing(self, x_test: np.array, y_test: np.array, n_samples_percent: float = 1.0) -> float:
        """test the model on a given dataset"""
        n_samples = int(len(x_test) * n_samples_percent)
        if n_samples_percent == 1.0:
            sample_idx = list(range(len(x_test)))
        else:
            sample_idx = np.random.choice(
                range(x_test.shape[0]), size=n_samples, replace=False
            )

        total_correct = 0
        for sample_idx in sample_idx:
            y = y_test[sample_idx]
            x = x_test[sample_idx][:]
            prediction = self.predict(x)

            if prediction == y:
                total_correct += 1

        accuracy = total_correct / len(x_test)
        return accuracy
