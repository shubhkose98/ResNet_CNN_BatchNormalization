# Skip Connections(Residual Networks) and Batch Normalization in Deep Neural Networks
Implementation of Skip Connections(also called Residual Connections or ResNets) in CNNs to solve the problem of vanishing gradients. The vanishing gradient problem occurs when training deep neural networks, and it becomes challenging for gradients to propagate through many layers, leading to slow learning and convergence.
Skip connections are also known as identity shortcuts and work by allowing the output from one layer to be added directly to the output of a layer that is not necessarily its immediate successor. The basic building block of skip connection is the residual block or skip block, which includes a shortcut connection. They allow the gradient to flow more easily during backpropagation by skipping one or more layers.
Skip connections and batch normalization are two majorly used methods to solve vanishing gradient descent problem.


## Implementation
- **SkipBlock Class**:
The SkipBlock class represents a residual block with optional skip connections. It consists of two convolutional layers, each followed by batch normalization and ReLU activation. The class supports channel dimension adjustments via a 1x1 convolution if the downsample flag is set. Skip connections enable the block to add the input to the output, allowing information to bypass layers, enhancing learning efficiency.

- The main class stacks multiple SkipBlock instances to create a deep hierarchical network. The architecture begins with a convolutional layer followed by max-pooling. Skip connections are 
  utilized to allow information flow across blocks. The network concludes with two fully connected layers (fc1 and fc2) for classification. The model’s depth and skip connection usage are 
  configurable, making it adaptable for various scenarios.

- **Batch Normalization**:
Batch normalization normalizes the activations of each layer, reducing internal covariate shift and providing a regularization effect. It stabilizes the training process, accelerates convergence, and allows for the use of higher learning rates.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- The PDF file attached contains all of the implementation documentation and libraries and datasets required.
