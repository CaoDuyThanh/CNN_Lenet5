import theano
import theano.tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


class CNNLayer:
    def __init__(self,
                 learningRate,
                 imageShape,
                 filterShape,
                 poolingShape = (2, 2)):
        # Set parameters
        self.LearningRate = learningRate
        self.ImageShape = imageShape
        self.FilterShape = filterShape
        self.PoolingShape = poolingShape

        # Create convolution layer
        convLayer = conv2d(
            input = inputConv,
            input_shape = imageShape,
            filters = filtersConv,
            filter_shape = filterShape,
            border_mode =  'full'
        )


        # Create pooling layer
        poolLayer = pool_2d(
            input = inputData,
            ds = poolingShape,
            mode = 'max',
            ignore_border = True
        )

        