import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


def get_accuracy(classifier: kr.Model, dataset: tf.data.Dataset):
    dataset = dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)

    correct = 0
    wrong = 0

    for data in dataset:
        real_images = data['images']
        real_labels = data['labels']

        logits = classifier(real_images)

        predicted_labels = tf.argmax(logits, axis=-1)

        wrong_count = tf.math.count_nonzero(real_labels - predicted_labels)
        correct_count = predicted_labels.shape[0] - wrong_count

        correct += correct_count
        wrong += wrong_count

    return correct / (wrong + correct)
