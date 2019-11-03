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