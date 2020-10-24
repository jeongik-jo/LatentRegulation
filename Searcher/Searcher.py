import tensorflow.keras as kr
import tensorflow as tf
import HyperParameters as HP


@tf.function
def _step(input_images, latent_vectors: tf.Variable, generator: kr.Model, discriminator: kr.Model):
    with tf.GradientTape() as tape:
        generated_images = generator(latent_vectors.value())

        data_difference_losses = HP.dif_function(input_images, generated_images)
        latent_regulation_losses = HP.regulation_function(
            train_vectors=HP.train_distribution_function([latent_vectors.shape[0], HP.sampling_size]),
            latent_vectors=latent_vectors,
            generator=generator,
            discriminator=discriminator)

        losses = data_difference_losses + HP.latent_regulation_loss_weight * latent_regulation_losses
    grads = tape.gradient(losses, [latent_vectors])

    return grads, losses, generated_images


#@tf.function
def search(input_images, generator: kr.Model, discriminator: kr.Model):
    min_latent_vectors = []
    min_losses = []
    min_generated_images = []

    for input_image in input_images:
        latent_vectors = tf.Variable(HP.latent_initialize_function([HP.latent_size, HP.latent_vector_dim]))

        for _ in range(HP.latent_gd_size):
            grads, losses, generated_images = _step(tf.tile(tf.expand_dims(input_image, axis=0), [HP.latent_size, 1, 1, 1]),
                                                    latent_vectors, generator, discriminator)
            HP.search_optimizer.apply_gradients(zip(grads, [latent_vectors]))

            if HP.use_resampling:
                latent_vectors.assign(HP.resampling_function(latent_vectors))

        min_index = tf.argmin(losses)
        min_latent_vectors.append(latent_vectors[min_index])
        min_losses.append(losses[min_index])
        min_generated_images.append(generated_images[min_index])

    return tf.convert_to_tensor(min_losses), tf.convert_to_tensor(min_generated_images), tf.convert_to_tensor(min_latent_vectors)
