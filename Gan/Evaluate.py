import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as HP
from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt
import os


@tf.function
def _get_feature_samples(generator: kr.Model, real_images: tf.Tensor):
    batch_size = real_images.shape[0]
    latent_vectors = HP.train_distribution_function([batch_size, HP.latent_vector_dim])
    fake_images = generator(latent_vectors)

    real_images = tf.concat([real_images for _ in range(3)], axis=-1)
    real_images = tf.image.resize(real_images, [299, 299])
    fake_images = tf.concat([fake_images for _ in range(3)], axis=-1)
    fake_images = tf.image.resize(fake_images, [299, 299])

    real_features = HP.inception_model(real_images)
    fake_features = HP.inception_model(fake_images)

    return real_features, fake_features


def get_features(generator: kr.Model, test_dataset: tf.data.Dataset):
    test_dataset = test_dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)

    real_features = []
    fake_features = []

    for test_batch in test_dataset:
        real_features_batch, fake_features_batch = _get_feature_samples(generator, test_batch['images'])
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = tf.concat(real_features, axis=0)
    fake_features = tf.concat(fake_features, axis=0)

    return real_features, fake_features


#@tf.function
def get_fid(generator: kr.Model, test_dataset: tf.data.Dataset):
    real_features, fake_features = get_features(generator, test_dataset)

    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


def save_graph(fids):
    if not os.path.exists('./gan_results'):
        os.makedirs('./gan_results')

    epochs = [i + 1 for i in range(len(fids))]

    plt.plot(epochs, fids)
    plt.xlabel('epochs')
    plt.ylabel('average fid')

    plt.savefig('./gan_results/fids.png')
    np.savetxt('./gan_results/fids.txt', np.array(fids), fmt='%f')
