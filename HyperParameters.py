import tensorflow as tf
import tensorflow.keras as kr
import Regulations
import Main
import scipy.stats as ss
import numpy as np

image_shape = [28, 28, 1]
discriminator_initial_filter_size = 128
generator_initial_conv_shape = [7, 7, 512]
latent_vector_dim = 256 # latent vector dimension
class_size = 10

inception_model = kr.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)
batch_size = 32 # batch size for train classifier or GAN
save_image_size = 16 # save image size for GAN


# Shared
epochs = 200 # epochs to train classifier or GAN

train_data_size = -1 # if -1, use all data
test_data_size = -1 # if -1, use all data

evaluate_model = False # if True, evaluate model
epochs_per_evaluate = 1 # epochs per evaluate

train_distribution_function = tf.random.normal # latent distribution function for GAN or latent code recovery
train_cdf = ss.norm.cdf # cumulative distribution function of latent distribution function
#train_distribution_function = lambda x: tf.random.uniform(x, minval=-1, maxval=1)
#train_cdf = lambda x: ss.uniform.cdf(x / 2.0 + 0.5)
def unn(shape):
    normal = tf.random.normal(shape)
    return tf.where(normal < 0, tf.random.uniform(shape, minval=-1, maxval=0), normal)
def unn_cdf(xs):
    below_zero = np.where(np.logical_and(-1 < xs, xs < 0), xs / 2.0 + 0.5, 0)
    over_zero = np.where(0.0 <= xs, ss.norm.cdf(xs).astype('float32'), 0.0)
    return below_zero + over_zero
#train_distribution_function = unn # uniform and normal distribution
#train_cdf = unn_cdf


# GAN train
discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)
generator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)


# classifier train
classifier_optimizer = tf.optimizers.Adam(learning_rate=0.00001)


# searcher
dif_function = lambda x, y: tf.reduce_mean(tf.abs(x - y), axis=[1, 2, 3]) # diff function for latent code recovery

latent_initialize_function = train_distribution_function # init function for laten code recovery

latent_gd_size = 200 # number to perform gradient descent
latent_size = 16 # latent code number per data

search_optimizer = None # must be None
search_optimizer_learning_rates = [0.0001, 0.001, 0.01] # learning rates for latent code recovery optimizer. Search sequentially with these learning rates

use_latent_regulation_loss = True # if True, use latent regulation loss
latent_regulation_loss_weight = None # must be None
latent_regulation_loss_weights = [0.01, 0.1, 1.0] # latent regulation loss weight for latent code recovery. Search sequentially with latent regulation loss weight

#regulation_function = Regulations.wasserstein_distances # wasserstein distance latent regulation loss
#regulation_function = Regulations.energy_distances # energy distance latent regulation loss
sampling_size = 10000 # sampling size to approximate latent distribution for statistical distance latent regulation loss

regulation_function = Regulations.z_score_square # z score square latent regulation loss
#regulation_function = Regulations.trick_discriminator # fool discriminator latent regulation loss


use_resampling = False # if True, use element resampling
resampling_function = Regulations.logistic_cutoff # stochastic element resampling with logistic cutoff function
#resampling_function = Regulations.truncated_normal_cutoff # stochastic element resampling with truncated normal cutoff function
#resampling_function = Regulations.boundary_resampling # stochastic element resampling with boundary resampling

resampling_hyperparameter = 0 # hyperparameter for element resampling. should be 1.0 with boundary resampling.

#main_function = Main.train_classifier # train classifier 
#main_function = Main.test_classifier # test classifier
#main function = Main.train_gan # train GAN
#main_function = Main.test_gan # test GAN
main_function = Main.save_searcher_results # perform latent code recovery and save results
#main_function = Main.test_searcher # print results of latent code recovery. must run after Main.save_searcher_results.


