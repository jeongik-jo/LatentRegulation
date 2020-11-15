import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


@tf.function
def _train(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        real_images = data['images']
        batch_size = real_images.shape[0]
        latent_vectors = HP.train_distribution_function([batch_size, HP.latent_vector_dim])

        fake_images = generator(latent_vectors, training=True)

        real_adversarial_values = tf.squeeze(discriminator(real_images, training=True))
        fake_adversarial_values = tf.squeeze(discriminator(fake_images, training=True))

        alpha = tf.random.uniform([batch_size, 1, 1, 1])
        inner_images = real_images * alpha + fake_images * (1.0 - alpha)
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(inner_images)
            inner_adversarial_values = discriminator(inner_images, training=True)
            score = tf.squeeze(inner_adversarial_values)

        gp_gradient = inner_tape.gradient(score, inner_images)
        gp_slope = tf.sqrt(tf.reduce_sum(tf.square(gp_gradient), axis=[1, 2, 3]))
        gp_loss = tf.square(gp_slope - 1)

        discriminator_loss = -real_adversarial_values + fake_adversarial_values + HP.gp_loss_weight * gp_loss
        generator_loss = -fake_adversarial_values

    HP.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    HP.generator_optimizer.apply_gradients(
        zip(tape.gradient(generator_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    del tape
    del inner_tape


def train(generator: kr.Model, discriminator: kr.Model, data: tf.data.Dataset):
    data = data.shuffle(10000).batch(HP.batch_size).prefetch(1)

    for batch in data:
        _train(generator, discriminator, batch)
