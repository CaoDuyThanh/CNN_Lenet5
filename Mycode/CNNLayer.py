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

        # Create filter
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -1.0, high = 1.0, size = self.FilterShape),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.b = theano.shared(
            numpy.asarray(
                rng.uniform(low = -1.0, high = 1.0, size = self.FilterShape[0]),
                dtype=theano.config.floatX
            ),
            borrow = True
        )

        # Create convolution layer
        outputConv = conv2d(
            input = self.Input,
            input_shape = self.InputShape,
            filters = self.W,
            filter_shape = self.FilterShape
        )

        # Create pooling layer
        poolOut = pool_2d(
            input = outputConv,
            ds = self.PoolingShape,
            ignore_border = True
        )

        self.Output = theano.tensor.tanh(poolOut + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.Params = [self.W, self.b]

