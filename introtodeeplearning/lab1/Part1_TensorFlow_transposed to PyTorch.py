#!/usr/bin/env python
# coding: utf-8

# <table align="center">
#   <td align="center"><a target="_blank" href="http://introtodeeplearning.com">
#         <img src="https://i.ibb.co/Jr88sn2/mit.png" style="padding-bottom:5px;" />
#       Visit MIT Deep Learning</a></td>
#   <td align="center"><a target="_blank" href="https://colab.research.google.com/github/aamini/introtodeeplearning/blob/master/lab1/Part1_TensorFlow.ipynb">
#         <img src="https://i.ibb.co/2P3SLwK/colab.png"  style="padding-bottom:5px;" />Run in Google Colab</a></td>
#   <td align="center"><a target="_blank" href="https://github.com/aamini/introtodeeplearning/blob/master/lab1/Part1_TensorFlow.ipynb">
#         <img src="https://i.ibb.co/xfJbPmL/github.png"  height="70px" style="padding-bottom:5px;"  />View Source on GitHub</a></td>
# </table>
# 
# 
# # Copyright Information
# 

# In[1]:


# Copyright 2021 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.
# 
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of 6.S191 must
# reference:
#
# © MIT 6.S191: Introduction to Deep Learning
# http://introtodeeplearning.com
#


# # Transpose lab examples from Tensorflow to Pytorch

# This is my intent as an exercise to move all materials provided by MIT to pytorch framework. I think it will help me to better understand both frameworks. All materials are provided by MIT and is under the MIT license. And this is entirely coming from MIT 6.S191.
# 
# Most of the time in markdown entries I will replace Tensorflow with Pytorch, and when necessarely I will transpose tensorflow concepts to pytorch ones.
# 
# All the original text is from MIT and credit is for © MIT 6.S191: Introduction to Deep Learning http://introtodeeplearning.com

# # Lab 1: Intro to PyTorch and Music Generation with RNNs
# 
# In this lab, you'll get exposure to using PyTorch and learn how it can be used for solving deep learning tasks. Go through the code and run each cell. Along the way, you'll encounter several ***TODO*** blocks -- follow the instructions to fill them out before running those cells and continuing.
# 
# 
# # Part 1: Intro to PyTorch
# 
# ## 0.1 Install PyTorch
# 
# PyTorch is a software library extensively used in machine learning. Here we'll learn how computations are represented and how to define a simple neural network in PyTorch. For all the labs in 6.S191 2021, we'll be using the PyTorch v1.4, which affords great flexibility and the ability to imperatively execute operations, just like in Python. You'll notice that PyTorch is quite similar to Python in its syntax and imperative execution. Let's install PyTorch and a couple of dependencies.
# 

# In[2]:


get_ipython().system('cat ../../env\\ \\ mit_6S191.txt')


# In[3]:


get_ipython().system('conda list|grep torch')


# In[4]:


import torch
import numpy as np

# Download and import the MIT 6.S191 package
# !pip install mitdeeplearning
import mitdeeplearning as mdl

import matplotlib.pyplot as plt


# ## 1.1 Why is PyTorch called PyTorch?
# 
# The answer is not obvious (not as in TensorFlow or Keras). Looks like a possible explanation in stackoverflow: https://stackoverflow.com/a/52708294/9922126
# 
# > it could be derived from SMOrch algorithm. Light + Smorch = Torch. And Py from Python of course. 
# 
# Anyway PyTorch is dealing with Tensors, which are data structures that you can think of as multi-dimensional arrays. Tensors are represented as n-dimensional arrays of base dataypes such as a string or integer -- they provide a way to generalize vectors and matrices to higher dimensions.
# 
# The ```shape``` of a Tensor defines its number of dimensions and the size of each dimension. The ```rank``` of a Tensor provides the number of dimensions (n-dimensions) -- you can also think of this as the Tensor's order or degree.

# ### scalar
# Let's first look at 0-d Tensors, of which a scalar is an example:

# #### from TensorFlow

# In[3]:


sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))


# #### to PyTorch

# In[7]:


#this is not possible to store non-scalar item in tensors with pytorch (contrary to tensorflow)
sport = torch.tensor('Tennis')


# In[5]:


number = torch.tensor(1.41421356237, dtype=torch.float64)

print("`number` is a {}-d Tensor".format(len(number.shape)))


# ### Vectors and lists 
# can be used to create 1-d Tensors:

# #### from TensorFlow

# In[4]:


sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))


# #### to PyTorch

# In[6]:


data = [3.141592, 1.414213, 2.71821]
numbers = torch.tensor(data, dtype=torch.float64)
print("`numbers` is a {}-d Tensor with shape: {}".format(len(numbers.shape), list(numbers.shape)))


# we can use `.size()` as an alternative to `.shape`

# In[7]:


numbers.size()


# ### 2-d (matrices) and higher-rank Tensors

# Next we consider creating 2-d (i.e., matrices) and higher-rank Tensors. For examples, in future labs involving image processing and computer vision, we will use 4-d Tensors. Here the dimensions correspond to the number of example images in our batch, image height, image width, and the number of color channels.

# #### from TensorFlow

# In[5]:


### Defining higher-order Tensors ###

'''TODO: Define a 2-d Tensor'''
matrix = tf.constant([[3.141592, 1.414213, 2.71821], [3.141592, 1.414213, 2.71821]], tf.float64) # TODO

assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(matrix).numpy() == 2


# In[6]:


'''TODO: Define a 4-d Tensor.'''
# Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3. 
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = tf.zeros((10, 256, 256, 3),tf.float64)# TODO

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"


# #### to PyTorch

# In[8]:


### Defining higher-order Tensors ###

matrix = torch.tensor([[3.141592, 1.414213, 2.71821], [3.141592, 1.414213, 2.71821]], dtype=torch.float64)

matrix


# In[9]:


type(matrix)


# In[10]:


len(matrix.shape)


# In[11]:


'''TODO: Define a 4-d Tensor.'''
# Use torch.zeros(shape) to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3. 
#   You can think of this as 10 images where each image is RGB 256 x 256.

images = torch.zeros((10, 256, 256, 3),dtype=torch.float64)# TODO

assert isinstance(images, torch.Tensor), "images must be a torch Tensor object"
assert len(images.shape) == 4, "images must be of rank 4"
assert list(images.shape) == [10, 256, 256, 3], "images is incorrect shape"


# As you have seen, the ```shape``` of a Tensor provides the number of elements in each Tensor dimension. The ```shape``` is quite useful, and we'll use it often. You can also use slicing to access subtensors within a higher-rank Tensor:

# In[12]:


row_vector = matrix[1]
column_vector = matrix[:,2]
scalar = matrix[1, 2]

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))


# ## 1.2 Computations on Tensors
# 
# A convenient way to think about and visualize computations in PyTorch is in terms of graphs. We can define this graph in terms of Tensors, which hold data, and the mathematical operations that act on these Tensors in some order. Let's look at a simple example, and define this computation using PyTorch:
# 
# ![alt text](img/pytorch_comp.png)

# ### from TensorFlow

# In[8]:


# Create the nodes in the graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them!
c1 = tf.add(a,b)
c2 = a + b # TensorFlow overrides the "+" operation so that it is able to act on Tensors
print(c1)
print(c2)


# ### to PyTorch

# In[13]:


# Create the nodes in the graph, and initialize values
a = torch.tensor(15)
b = torch.tensor(61)

# Add them!
c1 = torch.add(a,b)
c2 = a + b # PyTorch overrides the "+" operation so that it is able to act on Tensors
print(c1)
print(c2)


# Notice how we've created a computation graph consisting of PyTorch operations, and how  the output is a Tensor with value 76 -- we've just created a computation graph consisting of operations, and it's executed them and given us back the result.
# 
# Now let's consider a slightly more complicated example:
# 
# ![alt text](https://raw.githubusercontent.com/aamini/introtodeeplearning/master/lab1/img/computation-graph.png)
# 
# Here, we take two inputs, `a, b`, and compute an output `e`. Each node in the graph represents an operation that takes some input, does some computation, and passes its output to another node.
# 
# Let's define a simple function in PyTorch to construct this computation function:

# ### from TensorFlow

# In[9]:


### Defining Tensor computations ###

# Construct a simple computation function
def func(a,b):
    '''TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).'''
    a = tf.constant(a, tf.float64)
    b=tf.constant(b, tf.float64)
    c = tf.add(a, b)
    d = tf.subtract(b, tf.constant(1, tf.float64)) 
    e = tf.multiply(c, d) 
    return e


# Now, we can call this function to execute the computation graph given some inputs `a,b`:

# In[10]:


# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a,b)
print(e_out)


# ### to PyTorch

# In[14]:


### Defining Tensor computations ###

# Construct a simple computation function
def func(a,b):
    '''TODO: Define the operation for c, d, e (use torch.add, torch.sub, torch.mul).'''
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    c = torch.add(a, b)
    d = torch.sub(b, torch.tensor(1, dtype=torch.float64)) 
    e = torch.mul(c, d) 
    return e


# Now, we can call this function to execute the computation graph given some inputs `a,b`:

# In[15]:


# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a,b)
print(e_out)


# Notice how our output is a Tensor with value defined by the output of the computation, and that the output has no shape as it is a single scalar value.

# ## 1.3 Neural networks in PyTorch
# We can also define neural networks in PyTorch.
# 
# Let's first consider the example of a simple perceptron defined by just one dense layer: $ y = \sigma(Wx + b)$, where $W$ represents a matrix of weights, $b$ is a bias, $x$ is the input, $\sigma$ is the sigmoid activation function, and $y$ is the output. We can also visualize this operation using a graph: 
# 
# ![alt text](img/computation-graph-2-pytorch.png)
# 
# Neural networks can be constructed using the ``torch.nn`` package.
# 
# ``nn`` depends on ``autograd`` to define models and differentiate them.
# An ``nn.Module`` contains layers, and a method ``forward(input)`` that
# returns the ``output``.
# We will first define a ```Module``` to implement the simple perceptron defined above.

# ### basic from TensorFlow

# In[12]:


### Defining a network Layer ###

# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(OurDenseLayer, self).__init__()
    self.n_output_nodes = n_output_nodes

  def build(self, input_shape):
    d = int(input_shape[-1])
#     print(d)
    # Define and initialize parameters: a weight matrix W and bias b
    # Note that parameter initialization is random!
    self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) # note the dimensionality
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

  def call(self, x):
    '''TODO: define the operation for z (hint: use tf.matmul)'''
    z = tf.add(tf.matmul(x, self.W), self.b) # TODO

    '''TODO: define the operation for out (hint: use tf.sigmoid)'''
    y = tf.sigmoid(z) # TODO
    return y

# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = tf.constant([[1,2.]], shape=(1,2))
y = layer.call(x_input)

# test the output!
print(y.numpy())
mdl.lab1.test_custom_dense_layer_output(y)


# ### basic to PyTorch

# In[16]:


### Defining a network Layer ###
import torch.nn as nn
import torch.nn.functional as F
# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

class OurDenseLayer(nn.Module):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
#         self.fc = nn.Linear(2, n_output_nodes)

    def build(self, input_shape):
        d = int(input_shape[-1])
        
        self.W = nn.Parameter(torch.rand(d, self.n_output_nodes))
        self.b = nn.Parameter(torch.rand(1, self.n_output_nodes))

    def forward(self, x):
        x = torch.matmul(x,self.W)+self.b
        x = torch.sigmoid(x)
        return x    
    
torch.manual_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = torch.tensor([[1,2.]])
y = layer(x_input)

# test the output!
print(y.detach().numpy())


# ### Sequential, Dense from Keras -  from TensorFlow

# Conveniently, TensorFlow has defined a number of ```Layers``` that are commonly used in neural networks, for example a [```Dense```](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable). Now, instead of using a single ```Layer``` to define our simple neural network, we'll use the  [`Sequential`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Sequential) model from Keras and a single [`Dense` ](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Dense) layer to define our network. With the `Sequential` API, you can readily create neural networks by stacking together layers like building blocks. 

# In[13]:


### Defining a neural network using the Sequential API ###

# Import relevant packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the number of outputs
n_output_nodes = 3

# First define the model 
model = Sequential()

'''TODO: Define a dense (fully connected) layer to compute z'''
# Remember: dense layers are defined by the parameters W and b!
# You can read more about the initialization of W and b in the TF documentation :) 
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable
dense_layer = Dense(n_output_nodes) # TODO

# Add the dense layer to the model
model.add(dense_layer)


# That's it! We've defined our model using the Sequential API. Now, we can test it out using an example input:

# In[14]:


# Test model with example input
x_input = tf.constant([[1,2.]], shape=(1,2))

'''TODO: feed input into the model and predict the output!'''
model_output = model.predict(x_input)# TODO
print(model_output)


# In addition to defining models using the `Sequential` API, we can also define neural networks by directly subclassing the [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=stable) class, which groups layers together to enable model training and inference. The `Model` class captures what we refer to as a "model" or as a "network". Using Subclassing, we can create a class for our model, and then define the forward pass through the network using the `call` function. Subclassing affords the flexibility to define custom layers, custom training loops, custom activation functions, and custom models. Let's define the same neural network as above now using Subclassing rather than the `Sequential` model.

# In[15]:


### Defining a model using subclassing ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubclassModel(tf.keras.Model):

  # In __init__, we define the Model's layers
  def __init__(self, n_output_nodes):
    super(SubclassModel, self).__init__()
    '''TODO: Our model consists of a single Dense layer. Define this layer.''' 
    self.dense_layer = Dense(n_output_nodes)

  # In the call function, we define the Model's forward pass.
  def call(self, inputs):
    return self.dense_layer(inputs)


# Just like the model we built using the `Sequential` API, let's test out our `SubclassModel` using an example input.
# 
# 

# In[16]:


n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))

print(model.call(x_input))


# Importantly, Subclassing affords us a lot of flexibility to define custom models. For example, we can use boolean arguments in the `call` function to specify different network behaviors, for example different behaviors during training and inference. Let's suppose under some instances we want our network to simply output the input, without any perturbation. We define a boolean argument `isidentity` to control this behavior:

# In[17]:


### Defining a model using subclassing and specifying custom behavior ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class IdentityModel(tf.keras.Model):
    # As before, in __init__ we define the Model's layers
    # Since our desired behavior involves the forward pass, this part is unchanged
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    '''TODO: Implement the behavior where the network outputs the input, unchanged, 
      under control of the isidentity argument.'''
    
    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        '''TODO: Implement identity behavior'''
        if (isidentity): return inputs
        return x


# Let's test this behavior:

# In[20]:


n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))
'''TODO: pass the input into the model and call with and without the input identity option.'''
out_activate = model.call(x_input, False) # TODO
out_identity = model.call(x_input, True) # TODO

print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))


# Now that we have learned how to define `Layers` as well as neural networks in TensorFlow using both the `Sequential` and Subclassing APIs, we're ready to turn our attention to how to actually implement network training with backpropagation.

# ### to PyTorch : nn vs nn.functional

# I will still have to develop my intuition about that.
# Some nice explanation in [PyTorch forum](https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/6)
# 
# In the following I am using `nn.Module`

# In[22]:


import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, n_input_node, n_output_nodes):
        super(Net, self).__init__()
        # dense
        self.fc = nn.Linear(n_input_node, n_output_nodes)

    def forward(self, x):
        x = self.fc(x)
        return x

n_output_nodes = 3
x_input = torch.tensor([1., 2.])
net = Net(x_input.shape[-1], n_output_nodes)

print(net(x_input))


# ## 1.4 Automatic differentiation
# 
# 
# ### from TensorFlow
# [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
# is one of the most important parts of TensorFlow and is the backbone of training with 
# [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). We will use the TensorFlow GradientTape [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape?version=stable) to trace operations for computing gradients later. 
# 
# When a forward pass is made through the network, all forward-pass operations get recorded to a "tape"; then, to compute the gradient, the tape is played backwards. By default, the tape is discarded after it is played backwards; this means that a particular `tf.GradientTape` can only
# compute one gradient, and subsequent calls throw a runtime error. However, we can compute multiple gradients over the same computation by creating a ```persistent``` gradient tape. 
# 
# First, we will look at how we can compute gradients using GradientTape and access them for computation. We define the simple function $ y = x^2$ and compute the gradient:

# In[21]:


### Gradient computation with GradientTape ###

# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
  # Define the function
  y = x * x
# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 6.0


# In training neural networks, we use differentiation and stochastic gradient descent (SGD) to optimize a loss function. Now that we have a sense of how `GradientTape` can be used to compute and access derivatives, we will look at an example where we use automatic differentiation and SGD to find the minimum of $L=(x-x_f)^2$. Here $x_f$ is a variable for a desired value we are trying to optimize for; $L$ represents a loss that we are trying to  minimize. While we can clearly solve this problem analytically ($x_{min}=x_f$), considering how we can compute this using `GradientTape` sets us up nicely for future labs where we use gradient descent to optimize entire neural network losses.

# In[23]:


### Function minimization with automatic differentiation and SGD ###

# Initialize a random value for our initial x
x = tf.Variable([tf.random.normal([1])])
print("Initializing x={}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the loss, 
#   compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(500):
    with tf.GradientTape() as tape:
        '''TODO: define the loss as described above'''
        loss = tf.math.square(x-x_f)# TODO

        # loss minimization using gradient tape
        grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
        new_x = x - learning_rate*grad # sgd update
        x.assign(new_x) # update the value of x
        history.append(x.numpy()[0])

# Plot the evolution of x as we optimize towards x_f!
plt.plot(history)
plt.plot([0, 500],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')


# `GradientTape` provides an extremely flexible framework for automatic differentiation. In order to back propagate errors through a neural network, we track forward passes on the Tape, use this information to determine the gradients, and then use these gradients for optimization using SGD.

# ### to PyTorch

# [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
# is one of the most important parts of PyTorch and is the backbone of training with 
# [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). We will use the PyTorch AutoGrad [`torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) to trace operations for computing gradients later. 
# 
# > Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.
# >
# > Training a NN happens in two steps:
# >
# > * **Forward Propagation**: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.
# > 
# > * **Backward Propagation**: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this [video from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).
# 
# First, we will look at how we can compute gradients using `autograd` and ``requires_grad=True`` and access them for computation. We define the simple function $y = x^2$ and compute the gradient:

# In[47]:


### Gradient computation with autograd ###

# y = x^2
# Example: x = 3.0
x = torch.tensor([3.], requires_grad=True)

y = x**2

external_grad = torch.tensor([1.])
y.backward(gradient=external_grad)

print(2*x == x.grad)


# In training neural networks, we use differentiation and stochastic gradient descent (SGD) to optimize a loss function. Now that we have a sense of how `autograd` can be used to compute and access derivatives, we will look at an example where we use automatic differentiation and SGD to find the minimum of $L=(x-x_f)^2$. Here $x_f$ is a variable for a desired value we are trying to optimize for; $L$ represents a loss that we are trying to  minimize. While we can clearly solve this problem analytically ($x_{min}=x_f$), considering how we can compute this using `autograd` sets us up nicely for future labs where we use gradient descent to optimize entire neural network losses.

# In[52]:


### Function minimization with automatic differentiation and SGD ###

# Initialize a random value for our initial x
x = torch.rand([1], requires_grad=True)
print("Initializing x={}".format(x.detach().numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the loss, 
#   compute the derivative of the loss with respect to x, and perform the SGD update.

external_grad = torch.tensor([1.])


for i in range(500):    
    loss = (x-x_f)**2
    loss.backward(gradient=external_grad)
    with torch.no_grad():
        x = x - learning_rate*x.grad
    x.requires_grad=True
    history.append(x.detach().numpy()[0])

# Plot the evolution of x as we optimize towards x_f!
plt.plot(history)
plt.plot([0, 500],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')


# In[ ]:




