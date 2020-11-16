import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


@tf.function
def _train(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        real_images = data['images']

        latent_vectors = HP.train_distribution_function([real_images.shape[0], HP.latent_vector_dim])
        fake_images = generator(latent_vectors, training=True)

        real_adversarial_values = discriminator(real_images, training=True)
        fake_adversarial_values = discriminator(fake_images, training=True)

        discriminator_loss = tf.reduce_mean(tf.square(real_adversarial_values - 1) + tf.square(fake_adversarial_values))
        generator_loss = tf.reduce_mean(tf.square(fake_adversarial_values - 1))

    HP.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    HP.generator_optimizer.apply_gradients(
        zip(tape.gradient(generator_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    del tape


def train(generator: kr.Model, discriminator: kr.Model, data: tf.data.Dataset):
    data = data.shuffle(10000).batch(HP.batch_size).prefetch(1)

    for batch in data:
        _train(generator, discriminator, batch)
