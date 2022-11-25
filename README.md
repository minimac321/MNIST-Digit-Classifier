# MNIST-TensorFlow
Solve the classical MNIST dataset by implementing a convolutional neural network multiclass image classification 


Neural Networks
Implement the following to build a Neural Network:
- Activation Functions
- Gradients
- Forward Propagation
- Back Propagation
- Derivative of loss w.r.t. weights
- Derivative of loss w.r.t. input
- Weight updates
- Full train cycle

Activation Functions
![Activation Functions](input/activation_functions.png)

## What is the vanishing gradient problem (with respect to sigmoid function and its gradient) ?
At points with large positive or negative values, the gradient is very close to zero. As we backpropegate the gradients from end to start of the hidden layers, our gradients generally go from large to small. This means that our initia hidden layers might have gradients of zero - which means no learning is occuring.

Root cause: Squeezing nature of sigmoid and arc-tan.

Solution: Use ReLU

However, make sure we use the ReLU which can handle negatives.