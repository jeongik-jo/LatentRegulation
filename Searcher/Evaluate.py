import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP
import Searcher.Searcher
import scipy.stats as ss
import numpy as np
import os


def get_accuracy(classifier: kr.Model):
    folder_name = './searcher_results/learning_rate_' + str(HP.search_optimizer.learning_rate.numpy())
    if HP.use_latent_regulation_loss:
        folder_name += '/' + HP.regulation_function.__name__ + str(HP.latent_regulation_loss_weight)
    else:
        folder_name += '/no_regulation_loss'
    if HP.use_resampling:
        folder_name += '/' + HP.resampling_function.__name__ + str(HP.resampling_hyperparameter)
    else:
        folder_name += '/no_resampling'

    generated_images = np.load(folder_name + '/generated_images.npy')
    real_labels = np.load(folder_name + '/real_labels.npy')

    dataset = tf.data.Dataset.from_tensor_slices({'images': generated_images, 'labels': real_labels}).batch(HP.batch_size).prefetch(1)

    correct = 0
    wrong = 0

    for data in dataset:
        generated_images = data['images']
        real_labels = data['labels']

        logits = classifier(generated_images)

        predicted_labels = tf.argmax(logits, axis=-1)

        wrong_count = tf.math.count_nonzero(real_labels - predicted_labels)
        correct_count = predicted_labels.shape[0] - wrong_count

        correct += correct_count
        wrong += wrong_count

    return correct / (wrong + correct)


def get_l1_l2_loss():
    folder_name = './searcher_results/learning_rate_' + str(HP.search_optimizer.learning_rate.numpy())
    if HP.use_latent_regulation_loss:
        folder_name += '/' + HP.regulation_function.__name__ + str(HP.latent_regulation_loss_weight)
    else:
        folder_name += '/no_regulation_loss'
    if HP.use_resampling:
        folder_name += '/' + HP.resampling_function.__name__ + str(HP.resampling_hyperparameter)
    else:
        folder_name += '/no_resampling'

    generated_images = np.load(folder_name + '/generated_images.npy')
    real_images = np.load(folder_name + '/real_images.npy')

    l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(generated_images - real_images), axis=[1, 2, 3]))
    l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(generated_images - real_images), axis=[1, 2, 3]))

    return l1_loss, l2_loss


def latent_ks_test():
    folder_name = './searcher_results/learning_rate_' + str(HP.search_optimizer.learning_rate.numpy())
    if HP.use_latent_regulation_loss:
        folder_name += '/' + HP.regulation_function.__name__ + str(HP.latent_regulation_loss_weight)
    else:
        folder_name += '/no_regulation_loss'
    if HP.use_resampling:
        folder_name += '/' + HP.resampling_function.__name__ + str(HP.resampling_hyperparameter)
    else:
        folder_name += '/no_resampling'

    latent_vectors = np.load(folder_name + '/latent_vectors.npy')

    mean = tf.reduce_mean(latent_vectors)
    variance = tf.reduce_mean(tf.square(latent_vectors)) - tf.square(tf.reduce_mean(latent_vectors))
    p_value = ss.kstest(tf.reshape(latent_vectors, [-1]), HP.train_cdf)[1]

    return p_value, mean, variance


def save_searcher_results(generator: kr.Model, discriminator: kr.Model, dataset: tf.data.Dataset):
    dataset = dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)

    generated_images_set = []
    latent_vectors_set = []
    real_labels_set = []
    real_images_set = []

    for data in dataset:
        real_images = data['images']
        real_labels = data['labels']
        _, generated_images, latent_vectors = Searcher.Searcher.search(real_images, generator, discriminator)

        generated_images_set.append(generated_images)
        latent_vectors_set.append(latent_vectors)
        real_labels_set.append(real_labels)
        real_images_set.append(real_images)

    folder_name = './searcher_results/learning_rate_' + str(HP.search_optimizer.learning_rate.numpy())
    if HP.use_latent_regulation_loss:
        folder_name += '/' + HP.regulation_function.__name__ + str(HP.latent_regulation_loss_weight)
    else:
        folder_name += '/no_regulation_loss'
    if HP.use_resampling:
        folder_name += '/' + HP.resampling_function.__name__ + str(HP.resampling_hyperparameter)
    else:
        folder_name += '/no_resampling'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.save(folder_name + '/generated_images.npy', tf.concat(generated_images_set, axis=0))
    np.save(folder_name + '/latent_vectors.npy', tf.concat(latent_vectors_set, axis=0))
    np.save(folder_name + '/real_labels.npy', tf.concat(real_labels_set, axis=0))
    np.save(folder_name + '/real_images.npy', tf.concat(real_images_set, axis=0))
