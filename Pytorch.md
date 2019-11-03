## PyTorch

Basic datastructure in PyTorch for neural networks is Tensors.
### Tensor
Tensors are the generalisation of the matrices.

1D Tensors: Vectors. <break>
2D Tensors: Matrices <break>
3D Tensors: Array with three-indices. Ex: RGB Color of images

## Generating tensors

- **torch.Tensor()** is just an alias to **torch.FloatTensor()** which is the default type of tensor, when no dtype is specified during tensor construction.
- **torch.randn()** : Creates a tensor (dimensions passed in arguments) with random normal values  
- **torch.randn_like()**: Creates a tensor (same shape as tensor passed in argument) and fills the values with random normal variables.

### Reshaping operations

**torch.randn(b,a)** is any tensor.

- **torch.randn(b,a).reshape(a,b)**: Returns a tensor with data copied to a clone and stored in another part of memory.
- **any-tensor.resize_(a,b)**: returns the same tensor with a different shape.  However, if the new shape results in fewer elements than the original tensor, some elements will be removed from the tensor (but not from memory). If the new shape results in more elements than the original tensor, new elements will be uninitialized in memory. Underscore means it's a **In-place operation**. <br>
        An in-place operation is an operation that changes directly the content of a given Tensor without making a copy. Inplace operations in pytorch are always postfixed with a _, like .add_() or .scatter_(). Python operations like += or *= are also inplace operations.
- **any-tensor.view(a,b)** : will return a new tensor with the same data as weights with size (a, b).
     

### Tensor operations:

- **tensor.sum()** : Tensors have (a+b).sum() operation.  
- **tensor.sum(tensor, dim=1)** : For a 2D tensor, it takes sum across the columns. 
- **torch.exp()**
- **torch.manual_seed()** : Set random seed so things are predictable.
- **torch.flatten()** : Converts any tensor to 1D tensor.
 **NOTE** : Tensor is not same as torch!!

### Preferred operations:

- For multiplication: **torch.mm()**  or **torch.matmul()**  Runs on GPUs
- For reshaping: tensor.resize_() or tensor.view() (flatten input)

### Numpy operations: 
For data preprocessing.
- **torch.from_numpy()** : Creates a torch tensor from numpy array.  b = tensor.from_numpy(a)
- **tensor.numpy()** : converts a tensor to numpy array. b.numpy()

**NOTE**: The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.


### Broadcasting operations

- **numpy.broadcast_to()** 
- **tensor.gt(n)** : Returns a tensor of same shape, as one's and zeroes where 1's indicates element is greater than n, 0 otherwise. Applicable to **lt**, **eq**, **ge** etc.


## Neural Networks hacks

<strong> Fully Connected Layer </strong>

- Suppose **any-tensor** shape = \[a,b,c,d\], **any-tensor.view(a,-1)** or **any-tensor.view(a, b\*c\*d)** will flatten the 2nd, 3rd and 4th dimension into one single dimesion and return a 2D tensor. 

- Weight matrices is of dimension <strong>n*m</strong> where **n** is the number of input features and **m** is the number of nodes in the next layer.

<h3> nn Module </h3>

```python
class Network(nn.Module):
```

Here we're inheriting from `nn.Module`. Combined with `super().__init__()` this creates a class that tracks the architecture and provides a lot of useful methods and attributes. It is mandatory to inherit from `nn.Module` when you're creating a class for your network. The name of the class itself can be anything.

```python
self.hidden = nn.Linear(784, 256)
```

This line creates a module for a linear transformation, $x\mathbf{W} + b$, with 784 inputs and 256 outputs and assigns it to `self.hidden`. The module automatically creates the weight and bias tensors which we'll use in the `forward` method. You can access the weight and bias tensors once the network (`net`) is created with `net.hidden.weight` and `net.hidden.bias`.

```python
self.output = nn.Linear(256, 10)
```

Similarly, this creates another linear transformation with 256 inputs and 10 outputs.

```python
self.sigmoid = nn.Sigmoid()
self.softmax = nn.Softmax(dim=1)
```

Here I defined operations for the sigmoid activation and softmax output. Setting `dim=1` in `nn.Softmax(dim=1)` calculates softmax across the columns.

```python
def forward(self, x):
```

PyTorch networks created with `nn.Module` must have a `forward` method defined. It takes in a tensor `x` and passes it through the operations you defined in the `__init__` method.

```python
x = self.hidden(x)
x = self.sigmoid(x)
x = self.output(x)
x = self.softmax(x)
```

Here the input tensor `x` is passed through each operation and reassigned to `x`. We can see that the input tensor goes through the hidden layer, then a sigmoid function, then the output layer, and finally the softmax function. It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build. The order in which you define things in the `__init__` method doesn't matter, but you'll need to sequence the operations correctly in the `forward` method.

Now we can create a `Network` object.

model = Network()

Modifying the weights and biases of `Network` object:
<strong>model.hidden.bias.data.fill_(0)</strong> : To modify the biases.
<strong>model.hidden.weight.data.normal_(std=0.01)</strong> : sample from random normal with standard dev = 0.01


### nn.Sequential

nn.Sequential is a platform for convenient way for passing tensor sequentially in an Neural network.

Example of simple NN:

```
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
```