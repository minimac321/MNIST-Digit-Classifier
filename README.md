# MNIST-TensorFlow
Solve the classical MNIST multiclass image classification problem by creating a Feed Forward
Neural Network from scratch.

Methods Implemented:
- Activation Functions
- Gradients
- Forward Propagation
- Back Propagation
- Derivative of loss w.r.t. weights
- Derivative of loss w.r.t. input
- Weight updates
- Full train cycle


Activation Functions
<img src="input/activation_functions.png" alt="activation_funcs" title="Activation functions" width="400" height="200" /> 


## MNIST Solution:
Problem type: Classification (10 classes)

Neural Network loss function: Cross Entropy  


#### Neural Network Structure
Create a Neural Network which have the following layers:
- Input: **784** nodes with a Relu activation function
- Hidden Layer: **300** Nodes with a Relu activation function
- Output Layer: **10** Nodes with softmax activation function

#### Training Data
<img src="input/mnist_digits.png" alt="Kitten" title="MNIST training data" width="450" height="400" /> 


#### Training 
|    Training Loss wrt Iterations    | Training and Validation Accuracy |
|:----------------------------------:|:--------------------------------:|
|                                    |                                  |
| ![](input/loss_wrt_iterations.png) |  ![](input/train_vs_valid.png)   |


#### Predictions 
Example of an image with the true label of 2 and a predicted label of 2.
<img src="input/predicted_label.png" alt="predicted" title="MNIST predicted label" width="450" height="400" /> 


#### Accuracy
Final Test Set accuracy is **97.1%**
