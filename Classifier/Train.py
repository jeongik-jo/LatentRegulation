import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


@tf.function
def _train(classifier: kr.Model, data: tf.Tensor):
    with tf.GradientTape() as tape:
        real_images = data['images']
        real_labels = tf.one_hot(data['labels'], HP.class_size)

        real_logits = classifier(real_images, training=True)
        loss = tf.losses.categorical_crossentropy(real_labels, real_logits)

    HP.classifier_optimizer.apply_gradients(
        zip(tape.gradient(loss, classifier.trainable_variables),
            classifier.trainable_variables)
    )


def train(classifier: kr.Model, data: tf.data.Dataset):
    data = data.shuffle(10000).batch(HP.batch_size).prefetch(1)

    for batch in data:
        _train(classifier, batch)
