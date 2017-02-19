import theano
import theano.tensor as T
import numpy
from LogisticSGD import LogisticRegression, LoadData
from CNNLayer import CNNLayer
from HiddenLayer import HiddenLayer

def evaluateLenet5(datasetName = 'minist.pkl.gz',
                   batchSize = 500,
                   nEpochs = 200):
    # Random state
    rng = numpy.random.RandomState(22323);

    # Read data
    datasets = LoadData(datasetName)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX, testSetY = datasets[2]

    nTrainBatches = trainSetX.get_value(borrow=True).shape[0]
    nValidBatches = validSetX.get_value(borrwo=True).shape[0]
    nTestBatches = testSetX.get_value(borrow=True).shape[0]

    nTrainBatches //= batchSize
    nValidBatches //= batchSize
    nTestBatches //= batchSize

    # Create model
    X = T.matrix('X', dtype = theano.config.floatX)
    Y = T.vector('Y', dtype = theano.config.floatX)

    print ('Building the model...')
    nkerns = [[6, 5, 5],
              [16, 5, 5]]

    layer0Input = X.reshape((batchSize, 1, 32, 32))
    # Create first layer - CNN
    layer0 = CNNLayer(
        rng = rng,
        input = layer0Input,
        inputShape = (batchSize, 1, 32, 32),
        filterShape = (nkerns[0][0], 1, nkerns[0][1], nkerns[0][2])
    )

    # Create second layer - CNN
    layer1 = CNNLayer(
        rng = rng,
        input = layer0.Output,
        inputShape = (batchSize, nkerns[0][0], layer0.Output.shape[2], layer0.Output.shape[3]),
        filterShape = (nkerns[1][0], nkerns[0][0], nkerns[1][1], nkerns[1][2])
    )
    layer1Output = layer1.Output.flatten(2)

    # Create third layer - Fully Connected
    layer2 = HiddenLayer(
        rng = rng,
        input = ?,
        numNeurons = ?,
        numOut = ?,
        activaion = T.tanh
    )

    # Create forth layer - Fully Connected
    layer3 = HiddenLayer(

    )

    # Create fifth layer = Gaussian Connection
    layer4 = GaussLayer(

    )

    # Calculate cost function


    # Create train model
    trainModel = theano.function(

    )

    # Create valid model
    validModel = theano.function(

    )

    # Create test model
    testModel = theano.function(

    )

    print ('Building the model...Done')

    # Train model
    print ('Training the model...')


    # Gradient descent


    print ('Training the model...Done')


if __name__ == '__main__':
    evaluateLenet5()