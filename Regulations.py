import tensorflow as tf
import HyperParameters as HP


#@tf.function
def _cdf_distances(p, u_values, v_values):
    u_sorter = tf.argsort(u_values, axis=1)
    v_sorter = tf.argsort(v_values, axis=1)

    all_values = tf.sort(tf.concat([u_values, v_values], axis=1), axis=1)

    deltas = all_values[:, 1:] - all_values[:, :-1]
    u_cdf_indices = tf.searchsorted(tf.gather(u_values, u_sorter, batch_dims=1), all_values[:, :-1], 'right')
    v_cdf_indices = tf.searchsorted(tf.gather(v_values, v_sorter, batch_dims=1), all_values[:, :-1], 'right')

    u_cdf = tf.cast(u_cdf_indices / u_values.shape[1], dtype='float32')
    v_cdf = tf.cast(v_cdf_indices / v_values.shape[1], dtype='float32')

    if p == 1:
        return tf.reduce_sum(tf.abs(u_cdf - v_cdf) * deltas, axis=1)
    if p == 2:
        return tf.sqrt(tf.reduce_sum(tf.square(u_cdf - v_cdf) * deltas, axis=1))


#@tf.function
def wasserstein_distances(train_vectors, latent_vectors, **kwargs):
    return _cdf_distances(1, train_vectors, latent_vectors)


#@tf.function
def energy_distances(train_vectors, latent_vectors, **kwargs):
    return tf.sqrt(2.0) * _cdf_distances(2, train_vectors, latent_vectors)


#@tf.function
def lukaszyk_karmowski_distances(train_vectors, latent_vectors, **kwargs):
    return tf.reduce_mean(tf.abs(tf.expand_dims(train_vectors, axis=-1) - tf.expand_dims(latent_vectors, axis=-2)), axis=[1, 2])


#@tf.function
def bhattacharyya_distances(latent_vectors, **kwargs):
    return tf.reduce_mean(-tf.math.log(tf.reduce_sum(tf.exp(-0.25*tf.square(latent_vectors)), axis=1)))


#@tf.function
def z_score_square(latent_vectors, **kwargs):
    return tf.reduce_mean(tf.square(latent_vectors), axis=1)


#@tf.function
def z_score_absolute(latent_vectors, **kwargs):
    return tf.reduce_mean(tf.abs(latent_vectors), axis=1)


#@tf.function
def trick_discriminator(latent_vectors, generator, discriminator, **kwargs):
    logits = discriminator(generator(latent_vectors))
    return tf.square(tf.reshape(logits, [-1]) - 1)


#@tf.function
def logistic_cutoff(latent_vectors, **kwargs):
    probabilities = 1.0 / (1.0 + tf.exp(-HP.resampling_hyperparameter * (tf.abs(latent_vectors) - 2)))
    return tf.where(
        probabilities > tf.random.uniform(latent_vectors.shape),
        HP.train_distribution_function(latent_vectors.shape),
        latent_vectors
    )


#@tf.function
def truncated_normal_cutoff(latent_vectors, **kwargs):
    probabilities = tf.exp(-tf.square(HP.resampling_hyperparameter)/2.0)
    probabilities = probabilities / tf.exp(-tf.square(latent_vectors)/2.0)
    probabilities = tf.where(tf.abs(latent_vectors) < HP.resampling_hyperparameter, probabilities, 1)

    return tf.where(
        probabilities > tf.random.uniform(latent_vectors.shape),
        HP.train_distribution_function(latent_vectors.shape),
        latent_vectors
    )


#@tf.function
def boundary_resampling(latent_vectors, **kwargs):
    return tf.where(tf.abs(latent_vectors) > HP.resampling_hyperparameter,
                    HP.train_distribution_function(latent_vectors.shape),
                    latent_vectors)
