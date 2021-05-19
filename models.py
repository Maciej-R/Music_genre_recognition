import tensorflow as tf
from math import floor


def make_model(shape, name):
    if (name == "test"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=3,
                                   activation='relu',
                                   input_shape=shape,
                                   name='conv1'),

            tf.keras.layers.MaxPooling1D(name='max1'),

            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=3,
                                   activation='relu',
                                   name='conv2'),

            tf.keras.layers.MaxPooling1D(name='max2'),

            tf.keras.layers.Flatten(name='flatten'),

            tf.keras.layers.Dense(100, activation='relu', name='dense1'),
            tf.keras.layers.Dropout(0.5, name='dropout2'),
            tf.keras.layers.Dense(20, activation='relu', name='dense2'),
            tf.keras.layers.Dropout(0.5, name='dropout3'),
            tf.keras.layers.Dense(10, name='dense3'),
        ])

    if (name == "recurrent1"):  # Szybciej dzieci będziecie mieć niż wytrenujecie to z tym embeddingiem
        # Three parameters below are modifiable
        strides = 2
        p_size = 2
        e_len = 10
        out = int((shape[1] - p_size) / strides) + 1
        reshape = out * shape[0]
        return tf.keras.Sequential([
            tf.keras.layers.Reshape((*shape, 1)),
            tf.keras.layers.MaxPooling2D((1, p_size), (1, strides), input_shape=shape),
            tf.keras.layers.Reshape((1, reshape)),
            tf.keras.layers.Embedding(65536, e_len),
            tf.keras.layers.Reshape((shape[0], int(reshape * e_len / shape[0]))),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256)),
            tf.keras.layers.Dense(10, name='dense3'),
        ])
