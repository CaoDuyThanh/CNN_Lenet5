from __future__ import print_function
import timeit
import sys
import Utils.DataHelper as DataHelper
import Utils.CostFHelper as CostFHelper
from Layers.HiddenLayer import *
from Layers.SoftmaxLayer import *


# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.01
NUM_EPOCH = 1000
BATCH_SIZE = 20
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 500

def evaluateMLP():
    # Load datasets from local disk or download from the internet
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX, testSetY = datasets[2]

    nTrainBatchs = trainSetX.get_value(borrow = True).shape[0]
    nValidBatchs = validSetX.get_value(borrow = True).shape[0]
    nTestBatchs = testSetX.get_value(borrow = True).shape[0]
    nTrainBatchs //= BATCH_SIZE
    nValidBatchs //= BATCH_SIZE
    nTestBatchs //= BATCH_SIZE

    # Create model
    '''
    MODEL ARCHITECTURE
    INPUT     ->    HIDDEN LAYER    ->    OUTPUT (Softmax + L2 regularization)
    (28x28)         (500 neurons)                    (10 neurons)
    '''
    # Create random state
    rng = numpy.random.RandomState(12345)

    # Create shared variable for input
    Index = T.lscalar('Index')
    X = T.matrix('X', dtype = theano.config.floatX)
    Y = T.ivector('Y')

    hidLayer0 = HiddenLayer(
        rng = rng,
        input = X,
        numIn = 28 * 28,
        numOut = 500,
        activation = T.tanh
    )
    hidLayer0Output = hidLayer0.Output()
    hidLayer0Params = hidLayer0.Params()

    hidLayer1 = HiddenLayer(
        rng = rng,
        input = hidLayer0Output,
        numIn = 500,
        numOut = 10,
        activation = T.tanh
    )
    hidLayer1Output = hidLayer1.Output()
    hidLayer1Params = hidLayer1.Params()

    # Softmax layer
    softmaxLayer0 = SoftmaxLayer(
        input = hidLayer1Output
    )
    softmaxLayer0Output = softmaxLayer0.Output()

    # Evaluate model - using early stopping
    # Define cost function = Regularization + Cross entropy of softmax
    costTrain = CostFHelper.CrossEntropy(softmaxLayer0Output, Y)
              #+ CostFHelper.L2(hidLayer0.W) + CostFHelper.L2(hidLayer1.W)
    hidLayer0GradW = T.grad(costTrain, hidLayer0Params[0])
    hidLayer0Gradb = T.grad(costTrain, hidLayer0Params[1])
    hidLayer1GradW = T.grad(costTrain, hidLayer1Params[0])
    hidLayer1Gradb = T.grad(costTrain, hidLayer1Params[1])
    updates = [
        (hidLayer0.W, hidLayer0.W - LEARNING_RATE * hidLayer0GradW),
        (hidLayer0.b, hidLayer0.b - LEARNING_RATE * hidLayer0Gradb),
        (hidLayer1.W, hidLayer1.W - LEARNING_RATE * hidLayer1GradW),
        (hidLayer1.b, hidLayer1.b - LEARNING_RATE * hidLayer1Gradb)
    ]
    error = CostFHelper.Error(softmaxLayer0Output, Y)

    # Train model
    trainModel = theano.function(
        inputs = [Index],
        outputs=costTrain,
        updates=updates,
        givens = {
            X: trainSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: trainSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    # Valid model
    validModel = theano.function(
        inputs = [Index],
        outputs= error,
        givens = {
            X: validSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: validSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    # Test model
    testModel = theano.function(
        inputs = [Index],
        outputs= error,
        givens = {
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
                    if (err <  best_error * IMPROVEMENT_THRESHOLD):
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
    evaluateMLP()
