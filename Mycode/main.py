import theano
import theano.tensor as T
import numpy
from LogisticSGD import LogisticRegression, LoadData
from CNNLayer import CNNLayer
from HiddenLayer import HiddenLayer
from SoftmaxLayer import SoftmaxLayer

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

def evaluateLenet5(datasetName = '../Dataset/mnist.pkl.gz',
                   learningRate = 0.005,
                   batchSize = 500,
                   nEpochs = 2000):
    # Random state
    rng = numpy.random.RandomState(22323);

    # Read data
    datasets = LoadData(datasetName)

    trainSetX, trainSetY = datasets[0]
    trainSetX = padData(trainSetX)
    validSetX, validSetY = datasets[1]
    validSetX = padData(validSetX)
    testSetX, testSetY = datasets[2]
    testSetX = padData(testSetX)

    nTrainBatches = trainSetX.get_value(borrow = True).shape[0]
    nValidBatches = validSetX.get_value( borrow = True).shape[0]
    nTestBatches = testSetX.get_value(borrow = True).shape[0]

    nTrainBatches //= batchSize
    nValidBatches //= batchSize
    nTestBatches //= batchSize

    # Create model
    X = T.matrix('X')
    Y = T.ivector('Y')

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
        inputShape = (batchSize, nkerns[0][0], 14, 14),
        filterShape = (nkerns[1][0], nkerns[0][0], nkerns[1][1], nkerns[1][2])
    )
    layer1Output = layer1.Output.flatten(2)

    # Create third layer - Fully Connected
    layer2 = HiddenLayer(
        rng = rng,
        input = layer1Output,
        numIn = 400,
        numOut = 120,
        activation = T.tanh
    )
    layer2Output = layer2.Output

    # Create forth layer - Fully Connected
    layer3 = HiddenLayer(
        rng = rng,
        input = layer2Output,
        numIn = 120,
        numOut = 84,
        activation = T.tanh
    )
    layer3Output = layer3.Output

    # Create fifth layer = Fully Connected
    layer4 = HiddenLayer(
        rng = rng,
        input = layer3Output,
        numIn = 84,
        numOut = 10,
        activation = T.tanh
    )
    layer4Output = layer4.Output

    # Create softmax layer
    layerSoftmax = SoftmaxLayer(
        input = layer4Output
    )
    output = layerSoftmax.Output

    # Calculate cost function
    cost = -T.mean(T.log(output)[T.arange(Y.shape[0]), Y])

    # Gradient
    params = layer4.Params + layer3.Params + layer2.Params + layer1.Params + layer0.Params
    grads = T.grad(cost, params)

    # Update
    updates = [
        (param_i, param_i - learningRate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # Create train model
    index = T.lscalar()
    trainModel = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            X: trainSetX[index * batchSize : (index + 1) * batchSize],
            Y: trainSetY[index * batchSize : (index + 1) * batchSize]
        }
    )

    # Calculate error function
    outPred = T.argmax(output, axis = 1);
    error = T.mean(T.neq(outPred, Y));

    # Create valid model
    validModel = theano.function(
        [index],
        error,
        givens = {
            X: validSetX[index * batchSize : (index + 1) * batchSize],
            Y: validSetY[index * batchSize : (index + 1) * batchSize],
        }
    )

    # Create test model
    testModel = theano.function(
        [index],
        error,
        givens={
            X: testSetX[index * batchSize: (index + 1) * batchSize],
            Y: testSetY[index * batchSize: (index + 1) * batchSize],
        }
    )

    print ('Building the model...Done')

    # Train model
    print ('Training the model...')
    epoch = 0
    while (epoch < nEpochs):
        epoch += 1
        for minibatchIndex in range(nTrainBatches):
            iter = (epoch - 1) * nTrainBatches + minibatchIndex

            costIJ = trainModel(minibatchIndex)

            if iter % 100 == 0:
                print ('Training %i iter = ', iter)

        # Test model
        errorValid = 0;
        for minibatchIndex in range(nValidBatches):
            errorValid += validModel(minibatchIndex)
        errorValid /= nValidBatches;
        print ('Error valid %i = ', errorValid)

        errorTest = 0;
        for minibatchIndex in range(nTestBatches):
            errorTest += testModel(minibatchIndex)
        errorTest /= nTestBatches;
        print ('Error valid %i = ', errorTest)

        # Gradient descent


    print ('Training the model...Done')


if __name__ == '__main__':
    evaluateLenet5()