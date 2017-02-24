from __future__ import print_function
import timeit
import sys
import Utils.DataHelper as DataHelper
import Utils.CostFHelper as CostFHelper
from Layers.HiddenLayer import *
from Layers.SoftmaxLayer import *
from Layers.ConvPoolLayer import *

# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.005
NUM_EPOCH = 1000
BATCH_SIZE = 20
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 500

def padData(sharedData):
    data = sharedData.get_value()
    numSamples = data.shape[0]
    data = data.reshape((numSamples, 28, 28))

    newData = numpy.zeros((numSamples, 32, 32))
    newData[:, 2 : 30, 2 : 30] = data
    newData = newData.reshape((numSamples, 32 * 32))
    return theano.shared(
        numpy.asarray(
            newData,
            dtype = theano.config.floatX
        ),
        borrow = True
    )

def evaluateLenet5():
    # Load datasets from local disk or download from the internet
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX, testSetY = datasets[2]

    trainSetX = padData(trainSetX)
    validSetX = padData(validSetX)
    testSetX = padData(testSetX)

    nTrainBatchs = trainSetX.get_value(borrow=True).shape[0]
    nValidBatchs = validSetX.get_value(borrow=True).shape[0]
    nTestBatchs = testSetX.get_value(borrow=True).shape[0]
    nTrainBatchs //= BATCH_SIZE
    nValidBatchs //= BATCH_SIZE
    nTestBatchs //= BATCH_SIZE

    # Create model
    '''
    MODEL ARCHITECTURE
    INPUT     ->    Convolution      ->        Dropout
    (32x32)        (6, 1, 5, 5)              (6, 14, 14)
              ->    Convolution      ->        Dropout
                   (16, 6, 5, 5)             (16, 5, 5)
              ->    Hidden layer
                    (120 neurons)
              ->    Hidden layer
                    (84 neurons)
              ->    Output layer (Softmax)
                    (10 neurons)
    '''
    # Create random state
    rng = numpy.random.RandomState(12345)

    # Create shared variable for input
    Index = T.lscalar('Index')
    X = T.matrix('X')
    Y = T.ivector('Y')

    X4D = X.reshape((BATCH_SIZE, 1, 32, 32))
    # Convolution & pooling layer 0
    convPoolLayer0 = ConvPoolLayer(
        rng = rng,
        input = X4D,
        inputShape = (BATCH_SIZE, 1, 32, 32),
        filterShape = (6, 1, 5, 5),

    )
    convPoolLayer0Output = convPoolLayer0.Output()
    convPoolLayer0Params = convPoolLayer0.Params()

    # Convolution & pooling layer 1
    convPoolLayer1 = ConvPoolLayer(
        rng = rng,
        input = convPoolLayer0Output,
        inputShape = (BATCH_SIZE, 6, 14, 14),
        filterShape = (16, 6, 5, 5)
    )
    convPoolLayer1Output = convPoolLayer1.Output()
    convPoolLayer1Params = convPoolLayer1.Params()
    convPoolLayer1OutputRes = convPoolLayer1Output.reshape((BATCH_SIZE, 16 * 5 * 5))

    # Hidden layer 0
    hidLayer0 = HiddenLayer(
        rng = rng,
        input = convPoolLayer1OutputRes,
        numIn = 16 * 5 * 5,
        numOut = 120,
        activation = T.tanh
    )
    hidLayer0Output = hidLayer0.Output()
    hidLayer0Params = hidLayer0.Params()

    # Hidden layer 1
    hidLayer1 = HiddenLayer(
        rng=rng,
        input=hidLayer0Output,
        numIn=120,
        numOut=84,
        activation=T.tanh
    )
    hidLayer1Output = hidLayer1.Output()
    hidLayer1Params = hidLayer1.Params()

    # Hidden layer 2
    hidLayer2 = HiddenLayer(
        rng = rng,
        input = hidLayer1Output,
        numIn = 84,
        numOut = 10,
        activation = T.tanh
    )
    hidLayer2Output = hidLayer2.Output()
    hidLayer2Params = hidLayer2.Params()

    # Softmax layer
    softmaxLayer0 = SoftmaxLayer(
        input=hidLayer2Output
    )
    softmaxLayer0Output = softmaxLayer0.Output()

    # List of params from model
    params = hidLayer2Params + \
             hidLayer1Params + \
             hidLayer0Params + \
             convPoolLayer1Params + \
             convPoolLayer0Params

    # Evaluate model - using early stopping
    # Define cost function = Regularization + Cross entropy of softmax
    costTrain = CostFHelper.CrossEntropy(softmaxLayer0Output, Y)

    # Define gradient
    grads = T.grad(costTrain, params)

    # Updates function
    updates = [
        (param, param - LEARNING_RATE * grad)
        for (param, grad) in zip(params, grads)
    ]

    # Train model
    trainModel = theano.function(
        inputs=[Index],
        outputs=costTrain,
        updates=updates,
        givens={
            X: trainSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: trainSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    error = CostFHelper.Error(softmaxLayer0Output, Y)
    # Valid model
    validModel = theano.function(
        inputs=[Index],
        outputs=error,
        givens={
            X: validSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: validSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    # Test model
    testModel = theano.function(
        inputs=[Index],
        outputs=error,
        givens={
            X: testSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: testSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    doneLooping = False
    iter = 0
    patience = PATIENCE
    best_error = 1
    best_iter = 0
    start_time = timeit.default_timer()
    epoch = 0
    while (epoch < NUM_EPOCH) and (not doneLooping):
        epoch = epoch + 1
        for indexBatch in range(nTrainBatchs):
            iter = (epoch - 1) * nTrainBatchs + indexBatch
            cost = trainModel(indexBatch)

            if iter % VALIDATION_FREQUENCY == 0:
                print ('Validate model....')
                err = 0;
                for indexValidBatch in range(nValidBatchs):
                    err += validModel(indexValidBatch)
                err /= nValidBatchs
                print ('Error = ', err)

                if (err < best_error):
                    if (err < best_error * IMPROVEMENT_THRESHOLD):
                        patience = max(patience, iter * PATIENCE_INCREASE)

                    best_iter = iter
                    best_error = err

                    # Test on test set
                    test_losses = [testModel(i) for i in range(nTestBatchs)]
                    test_score = numpy.mean(test_losses)

        if (patience < iter):
            doneLooping = True
            break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_error * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



if __name__ == "__main__":
    evaluateLenet5()