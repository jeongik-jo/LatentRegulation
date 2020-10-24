import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = kr.datasets.mnist.load_data()

    train_images = tf.cast(train_images / 127.5 - 1, dtype='float32')
    test_images = tf.cast(test_images / 127.5 - 1, dtype='float32')
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)

    train_labels = tf.cast(train_labels, dtype='int64')
    test_labels = tf.cast(test_labels, dtype='int64')

    train_dataset = tf.data.Dataset.from_tensor_slices({'images': train_images, 'labels': train_labels}).shuffle(10000)
    test_dataset = tf.data.Dataset.from_tensor_slices({'images': test_images, 'labels': test_labels}).shuffle(10000)

    if HP.train_data_size != -1:
        train_dataset = train_dataset.take(HP.train_data_size)

    if HP.test_data_size != -1:
        test_dataset = test_dataset.take(HP.test_data_size)

    return train_dataset, test_dataset
