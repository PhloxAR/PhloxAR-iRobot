from __future__ import unicode_literals

from sklearn import datasets
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets.classification import ClassificationDataSet

# network constants
INPUT = 4
HIDDEN = 3
OUTPUT = 1
CLASSES = 3


# get classification dataset
def _get_classification_dataset():
    return ClassificationDataSet(INPUT, OUTPUT, nb_classes=CLASSES)


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset):
    classification_dataset = _get_classification_dataset()

    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0],
                                         supervised_dataset.getSample(n)[1])

    return classification_dataset


# split dataset with proportion
def _split_with_proportion(dataset, proportion):
    x, y = dataset.splitWithProportion(proportion)

    x = _convert_supervised_to_classification(x)
    y = _convert_supervised_to_classification(y)

    return x, y


# build a neural network
def build_network():
    # get iris data
    iris = datasets.load_iris()
    d, t = iris.data, iris.target

    # build dataset
    ds = _get_classification_dataset()
    for i in range(len(d)):
        ds.addSample(d[i], t[i])

    print("Dataset input: {}".format(ds['input']))
    print("Dataset output: {}".format(ds['target']))
    print("Dataset input length: {}".format(len(ds['input'])))
    print("Dataset output length: {}".format(len(ds['target'])))
    print("Dataset length: {}".format(len(ds)))
    print("Dataset input|output dimensions are {}|{}".format(ds.indim,
                                                             ds.outdim))

    # split dataset
    train_data, test_data = _split_with_proportion(ds, 0.70)

    print("Train Data length: {}".format(len(train_data)))
    print("Test Data length: {}".format(len(test_data)))

    # encode with one output neuron per class
    train_data._convertToOneOfMany()
    test_data._convertToOneOfMany()

    print("Train Data input|output dimensions are {}|{}".format(
        train_data.indim, train_data.outdim))
    print("Test Data input|output dimensions are {}|{}".format(test_data.indim,
                                                               test_data.outdim))

    # build network
    network = buildNetwork(INPUT, HIDDEN, CLASSES, outclass=SoftmaxLayer)

    # train network
    trainer = BackpropTrainer(network, dataset=train_data, momentum=0.1,
                              verbose=True, weightdecay=0.01)
    trainer.trainOnDataset(train_data, 500)

    print("Total epochs: {}".format(trainer.totalepochs))

    # test network
    output = network.activateOnDataset(test_data).argmax(axis=1)

    print("Percent error: {}".format(percentError(output, test_data['class'])))

    # return network
    return network


# classify data against neural network
def classify_data(network, data):
    output = network.activate(data).argmax(axis=0)

    print("Data classified as: {}".format(output))

    return output
