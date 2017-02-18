import theano
import theano.tensor
import numpy
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from pylab import *

class CNNLayer:
    def __init__(self,
                 rng,
                 input,
                 inputShape,
                 filterShape,
                 poolingShape = (2, 2)):
        # Set parameters
        self.Input = input
        self.InputShape = inputShape
        self.FilterShape = filterShape
        self.PoolingShape = poolingShape

        # Calculate filters size


        # Create filter
        self.filters = theano.shared(
            numpy.asarray(
                rng.uniform(low = ..., high = ..., size = self.FilterShape),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        # Create convolution layer
        outputConv = convLayer = conv2d(
            input = self.Input,
            input_shape = self.InputShape,
            filters = ,
            filter_shape = self.FilterShape
        )

        # Create pooling layer
        poolOut = pool_2d(
            input = outputConv,
            ds = self.PoolingShape,
            ignore_border = True
        )

        self.Output = theano.tensor.tanh(poolOut + )

        self.Params = [self.W, self.b]

