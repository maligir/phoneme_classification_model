# phoneme_classification_model

This creates a phoneme classification model.

## Overall structure

1. Load the provided dataset
2. Create the train and test data loader objects
3. Define the network architecture
4. Instantiate the model, a loss function, and optimizer
5. Train the model with stochastic gradient descent, iterating over the training dataset several times
6. Evaluate the model on the held-out test data

## Initial Data

• train_feats:Anumpyarrayofshape(44730,11,40).Thefirstdimensionindicatesthe number of training examples. The second dimension indicates the number of consecutive feature frames we’ve extracted for each example - one center frame in the middle of the phoneme, with 5 frames of context on both the left and the right for a total of 11 acoustic frames. The last dimension indicates the number of Mel frequency filter bins that each frame has, in this case 40.
• test_feats: A numpy array of shape (4773, 11, 40). The first dimension indicates the number of test examples, while the second and third dimensions are exactly the same as for the training features.
• train_labels: A numpy integer array of length 44730, where the 𝑖𝑡h entry indicates the ground-truth class label for the 𝑖𝑡h training example. The class labels here are only integer indices; to obtain the string label, you can retrieve the corresponding element in the phone_labels array.
• test_labels: A numpy integer array of the same format as train_labels, but with 4773 examples used for testing.
• phone_labels: A length 48 array of strings, where the 𝑖𝑡h entry is the name of the 𝑖𝑡h class.

## Modeling and Training

The model is created using the standard PyTorch model with some linear layers and ReLU functions. The in built functions are filled to accomodate the parameters above and layers created in the most standard form.

During the training loop for stochastic gradient descent, we iterate over the entire training dataset multiple times. Each of these iterations of the outer loop is called a training epoch. During each epoch, the inner loop cycles through small subsets of the training data called minibatches.
One of PyTorch’s most important features is the autograd module. This module keeps track of every operation performed on all torch tensor objects that are registered with it, which is accomplished by specifying the requires_grad=True keyword argument when the tensor is instantiated. When .backward() is called on a tensor, it backpropagates its gradient into the tensors that were involved in its computation history, calling .backward() on them which continues until there are no more gradients to be computed.
Next we perform the parameter update. PyTorch also provides the optim module, which contains implementations of all of the popular optimization algorithms used in deep learning. Then we create a SGD optimization object, and also registers our model’s parameters with that object by passing them as an argument to the model’s init function. Then we call .step() on the optimizer after the gradient has been computed.

The model is then evaluated using varying metrics and methods.
