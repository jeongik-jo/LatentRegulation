import HyperParameters as HP
import Gan.Main, Classifier.Main, Searcher.Main
import tensorflow as tf
import numpy as np
import time


def train_gan():
    Gan.Main.train_gan()


def test_gan():
    Gan.Main.test_gan()


def train_classifier():
    Classifier.Main.train_classifier()


def test_classifier():
    Classifier.Main.test_classifier()


def test_searcher():
    p_values = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])
    vector_means = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])
    vector_variances = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])
    accuracies = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])
    l1_losses = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])
    l2_losses = np.zeros([len(HP.search_optimizer_learning_rates), len(HP.latent_regulation_loss_weights)])

    for i, learning_rate in enumerate(HP.search_optimizer_learning_rates):
        for j, latent_regulation_loss_weight in enumerate(HP.latent_regulation_loss_weights):
            HP.search_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
            HP.latent_regulation_loss_weight = latent_regulation_loss_weight
            p_values[i][j], vector_means[i][j], vector_variances[i][j] = Searcher.Main.test_searcher_ks_test()
            accuracies[i][j] = Searcher.Main.test_searcher_accuracy()
            l1_losses[i][j], l2_losses[i][j] = Searcher.Main.test_searcher_l1_l2_loss()

    print('\naccuracy :')
    print(accuracies)
    print('\np_values:')
    print(p_values)
    print('\nvector_means:')
    print(vector_means)
    print('\nvector_variances:')
    print(vector_variances)
    print('\nl1_losses')
    print(l1_losses)
    print('\nl2_losses')
    print(l2_losses)


def save_searcher_results():
    for i, learning_rate in enumerate(HP.search_optimizer_learning_rates):
        for j, latent_regulation_loss_weight in enumerate(HP.latent_regulation_loss_weights):
            HP.search_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
            HP.latent_regulation_loss_weight = latent_regulation_loss_weight
            print('start')
            print('learning rate: ' + str(learning_rate))
            print('latent regulation loss weight: ' + str(latent_regulation_loss_weight))
            start = time.time()
            Searcher.Main.save_searcher_results()
            print('end. time: ' + str(time.time() - start)[:7] + ' sec')

    test_searcher()
