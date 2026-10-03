## NanoNet

A tiny neural net built entirely in Numpy.
No PyTorch, no Tensorflow, only raw code and math for learning


# What's Inside
- MLP(784, 128, 10) from scratch 
- Gets ~96% on MNIST
- Built to learn how a neural net actually works at its core: autograd engine, forward, backprop, gradient descent, Adam, softmax, cross-entropy; all in Numpy 

# Struture
- `nanonet/autograd.py`: Tensor engine with autograd
- `nanonet/model.py`: MLP + Adam, training uses cross-entropy loss
- `nanonet/optim.py`: Adam optimizer
- `nanonet/manual.py`: first version with manual backprop
