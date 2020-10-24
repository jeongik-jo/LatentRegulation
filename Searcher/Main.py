import Data
import Classifier.Model, Gan.Model
from . import Evaluate


def test_searcher_accuracy():
    classifier = Classifier.Model.Classifier()
    classifier.load()

    accuracy = Evaluate.get_accuracy(classifier.model)

    return accuracy


def test_searcher_l1_l2_loss():
    return Evaluate.get_l1_l2_loss()


def test_searcher_ks_test():
    p_value, mean, variance = Evaluate.latent_ks_test()

    return p_value, mean, variance


def save_searcher_results():
    generator = Gan.Model.Generator()
    generator.load()
    discriminator = Gan.Model.Discriminator()
    discriminator.load()
    _, test_dataset = Data.load_dataset()

    Evaluate.save_searcher_results(generator.model, discriminator.model, test_dataset)


