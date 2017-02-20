import theano
import theano.tensor as T
import numpy
from LogisticSGD import LogisticRegression, LoadData
from CNNLayer import CNNLayer
from HiddenLayer import HiddenLayer
from SoftmaxLayer import SoftmaxLayer

def evaluateLenet5(datasetName = 'minist.pkl.gz',
                   learningRate = 0.005,
                   batchSize = 500,
                   nEpochs = 200):
    x = T.matrix('x')
    y = T.ivector('y')

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
        input = layer1Output,
        numOut = 120,
        activaion = T.tanh
    )
    layer2Output = layer2.Ouput

    # Create forth layer - Fully Connected
    layer3 = HiddenLayer(
        rng = rng,
        input = layer2Output,
        numOut = 84,
        activation = T.tanh
    )
    layer3Output = layer3.Ouput

    # Create fifth layer = Fully Connected
    layer4 = HiddenLayer(
        rng = rng,
        input = layer3Output,
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
    cost = -T.mean(T.log(output)[T.arange(y.shape[0]), y])

    # Gradient
    params = layer0.Params + layer1.Params + layer2.Params + layer3.Params + layer4.Params + layerSoftmax.Params
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
            x: trainSetX[index * batchSize : (index + 1) * batchSize],
            y: trainSetY[index * batchSize : (index + 1) * batchSize]
        }
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
    epoch = 0
    while (epoch < nEpochs):
        epoch += 1


    # Gradient descent


    print ('Training the model...Done')


if __name__ == '__main__':
    evaluateLenet5()