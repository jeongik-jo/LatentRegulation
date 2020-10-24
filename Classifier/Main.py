import time
import HyperParameters as HP
import Data
from . import Evaluate, Model, Train


def train_classifier():
    classifier = Model.Classifier()
    train_dataset, test_dataset = Data.load_dataset()

    for epoch in range(HP.epochs):
        print('iter', epoch)
        start = time.time()
        Train.train(classifier.model, train_dataset)

        print('saving...')
        classifier.save()
        print('saved')

        if HP.evaluate_model and (epoch + 1) % HP.epochs_per_evaluate == 0:
            print('accuracy: ', Evaluate.get_accuracy(classifier.model, test_dataset))
        print('time: ', time.time() - start)

    if not HP.evaluate_model:
        print('accuracy: ', Evaluate.get_accuracy(classifier.model, test_dataset))


def test_classifier():
    classifier = Model.Classifier()
    classifier.load()

    _, test_dataset = Data.load_dataset()

    start = time.time()
    print('accuracy: ', Evaluate.get_accuracy(classifier.model, test_dataset))
    print('time: ', time.time() - start)
