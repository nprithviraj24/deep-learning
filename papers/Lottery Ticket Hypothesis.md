### The Lotter Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

By Jonathan Frankle and Micheal Carbin.

#### About:
 This paper is about pruning in neural network. Pruning is the process of reducing the 
 parameter counts of trained neural networks. In this proposed paper, they demonstrate an
 existence of smaller subnetworks that when trained from the start learn at least as fast
  their larger counterparts while reaching similar test accuracy.
  
##### The Lottery Ticket Hypothesis

``A randomly-initialized, dense neural network contains a subnetwork that is initialized such that - 
when trained in isolation - it can match the test accuracy of the original network after
 training for at most the same number of iterations. ``
 
 ### Experiment
 
 1. Randomly initialize a neural network.
 2. Tran the network for \[ j \] 