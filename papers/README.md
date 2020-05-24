#### Deep Learning
... from the lens of an amateur researcher.

##### Motivation
In this repository, I share my views on some of the most highlighted conference papers. It will also
 contain random papers that I found interesting and elusive.
 
### Papers
 
 1. [The Lottery Ticket Hypothesis](https://openreview.net/forum?id=rJl-b3RcF7) (Paper of the year ICLR 2019)
 2. [Photo-realistic Facial Texture Transfer](https://arxiv.org/pdf/1706.04306.pdf)
 3. [Energy Based Models (ICLR 2020, and Yann Lecun - 2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
 

<break >

### Interesting ideas

###### 1. [Deep double descend plot (from Ilya Sutskever)](https://arxiv.org/pdf/1912.02292.pdf)

![Source](https://openai.com/content/images/2019/12/modeldd.svg)


- X axis: ResNet18 Width Parameters
- Y axis: Test/Train Error

If we increase the size of neural network, and if we don't do early stopping, we will notice
 gradual decline in train/test error. But, empirically it is proven that at some point the
test error will increase (overfitting) if we are continuing the process of increasing the width of NN as it
is evident from the graph. It will continue to increase the error, until point where it stops 
increasing the error and then eventually error rate decreases (test and train error).

This behaviour might primatively told us that larger networks tend to overfit. But maybe we didn't
increase the size of the network enough that it would outperform the previous its smaller versions.

Applicable to model size, training time, dataset size.

```Overfitting: Model is senstive to small training changes.```

<!--
Double Descent Phenomena: 
As we increase the number of parameters in a neural network, the test error initially decreases, increases 
and just as the model is able to fit the train set, undrgoes a second descent.

Possible explanation:
If dataset has as many as degrees of freedom as  the model, as if there are possible one-to-one correspondence
-->

###### 2. Automatic Domain Randomization (ADR)

Generate progressively more difficult environment as the system learns (alternative for self-play)
 

<!--  ###### Personal Note
`` This is just a honest attempt to improve my writing skills both personally and academically. 
If you are reading this and if you have a personal suggestion, please feel free to reach out to me. ``
-->
