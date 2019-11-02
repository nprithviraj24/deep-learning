# Deep Learning
Asking the right questions.

## Contents
- Prerequisites
- Questions
- Additional resources
- [Pytorch](Pytorch.md)

## Prerequisites:
- Machine Learning: Definition and how it works. 
- Neural Network.


## What is it all about?
A repo entirely focused on deep learning. Tries to answer following questions.
- What is Deep Learning?
- Why do we need Deep learning?
- How do we practically use deep learning?
<br />
<strong>Domain Specific</strong>
<br />
- Difference between image classification and object detection. 
<br />
<br />
Apart from that, this repo will be constantly updated with any new trends in deep learning. I recommend follow experts in this domain. Andrew Ng, Andrian Rosebrock to name a few.
<br />

### What is deep learning?
It is quite safe to say Machine Learning approach towards building an intelligent system is close to being obsolete. Paraphrasing it, Machine learning requires a lot of input with selected features beforehand to begin with, which can be hectic and tiring job. Sure it can be used to develop intelligent systems which can automate lot of tasks.
<br />

Providing required parameters and selected features can be abysmal. 
    Deep learning, on the other hand, expects minimal input(s) from the user. It calculates and extracts by itself. It's like a system with a nimble brain.
<br />
#### Deep Learning is Neural Networks 
- It is infact a convolutional Neural Network which is aimed to solve computer vision and image classification problems.  
- Deep Learning is a self-taught learning and unsupervised feature extraction and learning.

### Why do we need Deep Learning?
Emphatic use of Deep learning is in computer vision and image classification problems. Before jumping onto that, let's understand image classification and object detection. 
<br />
There is a great upsurge in computer vision field whose most typical applications are autonomous vehicles, smart video surveillance, facial detection and various people counting applications, fast and accurate object detection systems. These systems invole not only recognizing and classifying every object in an image, but localising each by drawing the appropriate bounding box and around it. This makes object detection a significantly harder task than it is trasitional computer vision predecessorr, image classification.

### How do we practically use deep learning?
As professor Jason Brownlee suggests, there are eight different applications of deep learning. 
<br />
<strong>Objects classification in images:</strong>
<br />
    Probably the most common use of deep learning. Involves image classification as well as object detection. 
<strong>Automatic Handwriting Generation</strong>
<br />
This is a task where given a corpus of handwriting examples, generate new handwriting for a given word or phrase.

The handwriting is provided as a sequence of coordinates used by a pen when the handwriting samples were created. From this corpus the relationship between the pen movement and the letters is learned and new examples can be generated ad hoc.

What is fascinating is that different styles can be learned and then mimicked. I would love to see this work combined with some forensic hand writing analysis expertise. 

<strong>Automatic Text Generation</strong>
<br />
This is an interesting task, where a corpus of text is learned and from this model new text is generated, word-by-word or character-by-character.

The model is capable of learning how to spell, punctuate, form sentiences and even capture the style of the text in the corpus.

Large recurrent neural networks are used to learn the relationship between items in the sequences of input strings and then generate text. More recently LSTM recurrent neural networks are demonstrating great success on this problem using a character-based model, generating one character at time.


### Deep learning with GP-GPU programming.
Today's consumer CPUs have at most 8 cores. Server CPUs ranges from 4 to 24, and these cores support hyperthreading which can create 8 to 48 threads respectively. In neural network we apply almost same operations on different values of same array. Most of the operations that are computed in CPU can be run parallely and calculated independently then aggrerated thereafter. So we are optimizing this task,by using GP-GPU programming.  <br />
    This can be achieved by using libraries:
        - OpenCL
        - CUDA <br /> 
        and many more...
    To refer a resources to practice parallel programming this [link](https://www.gitbook.com/book/leonardoaraujosantos/opencl/details) will be helpful. Only prerequisite is you must know MATLAB. <br />

If you are geek and want to know more about this.. refer [this](http://vertex.ai/blog/bringing-deep-learning-to-opencl). This link is specific to OpenCL. 



## Additional Resources

- [Understand Neural Network](http://datathings.com/blog/post/neuralnet/)
- [Understand Convolutional Neural Network](https://brohrer.github.io/how_convolutional_neural_networks_work.html)