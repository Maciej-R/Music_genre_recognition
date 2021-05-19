import tensorflow as tf


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
