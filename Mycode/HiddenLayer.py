import theano
import numpy
import theano.tensor as T

class HiddenLayer:
    def __init__(self,
                 rng,
                 input,
                 numIn,
                 numOut,
                 activation = T.tanh):
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation

        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (numIn, numOut)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.b = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (numOut)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        self.Output = self.Activation(T.dot(self.W, self.Input) + self.b)

        self.params = [self.W, self.b]

