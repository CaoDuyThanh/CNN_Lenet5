import theano
import theano.tensor
import numpy
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

class ConvPoolLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data
                 inputShape,            # Shape of input = [batch size, channels, rows, cols]
                 filterShape,           # Shape of filter = [number of filters, channels, rows, cols]
                 poolingShape = (2, 2)  # Shape of pooling (2, 2) default
                 ):
        # Set parameters
        self.Input = input
        self.InputShape = inputShape
        self.FilterShape = filterShape
        self.PoolingShape = poolingShape

        # Create shared parameters for filters
        self.W = theano.shared(
            input = numpy.asarray(
                rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = self.FilterShape
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )




    def Params(self):
        return 0


    def Output(self):
        return 0