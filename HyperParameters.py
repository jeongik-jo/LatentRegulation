import tensorflow as tf
import tensorflow.keras as kr
import Regulations
import Main
import scipy.stats as ss
import numpy as np

image_shape = [28, 28, 1]
discriminator_initial_filter_size = 128
generator_initial_conv_shape = [7, 7, 512]
latent_vector_dim = 256
class_size = 10

inception_model = kr.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)
batch_size = 32
save_image_size = 16


# Shared
epochs = 50

train_data_size = -1
test_data_size = -1

evaluate_model = False
epochs_per_evaluate = 10

train_distribution_function = tf.random.normal
train_cdf = ss.norm.cdf
#train_distribution_function = lambda x: tf.random.uniform(x, minval=-1, maxval=1)
#train_cdf = lambda x: ss.uniform.cdf(x / 2.0 + 0.5)
def unn(shape):
    normal = tf.random.normal(shape)
    return tf.where(normal < 0, tf.random.uniform(shape, minval=-1, maxval=0), normal)
def unn_cdf(xs):
    below_zero = np.where(np.logical_and(-1 < xs, xs < 0), xs / 2.0 + 0.5, 0)
    over_zero = np.where(0.0 <= xs, ss.norm.cdf(xs).astype('float32'), 0.0)
    return below_zero + over_zero
#train_distribution_function = unn
#train_cdf = unn_cdf

# GAN train
gp_loss_weight = 0.1
discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)
generator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)


# classifier train
classifier_optimizer = tf.optimizers.Adam(learning_rate=0.00001)


# searcher
dif_function = lambda x, y: tf.reduce_mean(tf.abs(x - y), axis=[1, 2, 3])

latent_initialize_function = train_distribution_function

latent_gd_size = 200
latent_size = 16

search_optimizer = None
search_optimizer_learning_rates = [0.0001, 0.001, 0.01]

use_latent_regulation_loss = True
latent_regulation_loss_weight = None
latent_regulation_loss_weights = [0.01, 0.1, 1.0]

#regulation_function = Regulations.wasserstein_distances
#regulation_function = Regulations.energy_distances
#regulation_function = Regulations.lukaszyk_karmowski_distances
#regulation_function = Regulations.bhattacharyya_distances
sampling_size = 1000

#regulation_function = Regulations.z_score_absolute
regulation_function = Regulations.z_score_square
#regulation_function = Regulations.trick_discriminator


use_resampling = False
resampling_function = Regulations.logistic_cutoff
#resampling_function = Regulations.truncated_normal_cutoff
#resampling_function = Regulations.boundary_resampling

resampling_hyperparameter = 0



main_function = Main.train_classifier

