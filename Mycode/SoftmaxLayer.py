from theano.tensor.nnet.nnet import softmax

class SoftmaxLayer:
    def __init__(self,
                 input):
        self.Input = input

        # Create layer
        self.Output = softmax(self.Input)