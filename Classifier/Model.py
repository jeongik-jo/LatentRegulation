import tensorflow as tf
import tensorflow.keras as kr
import Layers
import os
import HyperParameters as HP
import tensorflow_addons as tfa


class Classifier(object):
    def __init__(self):
        model_output = model_input = kr.Input(shape=HP.image_shape)
        model_output = kr.layers.Conv2D(filters=HP.discriminator_initial_filter_size, kernel_size=[3, 3],
                                        padding='same', activation=tf.nn.leaky_relu, use_bias=False)(model_output)
        model_output = tfa.layers.InstanceNormalization()(model_output)

        for _ in range(2):
            model_output = Layers.HalfResolution(conv_depth=3)(model_output)

        model_output = kr.layers.Flatten()(model_output)
        output_logit = kr.layers.Dense(units=HP.class_size, activation='softmax')(model_output)

        self.model = kr.Model(model_input, output_logit)

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('models')

        self.model.save_weights('./models/classifier.h5')

    def load(self):
        self.model.load_weights('./models/classifier.h5')

