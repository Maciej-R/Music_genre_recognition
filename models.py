import tensorflow as tf
from settings import BATCH_SIZE


def make_model(shape, name):
    if (name == "test"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128,
                                   kernel_size=3,
                                   activation='relu',
                                   input_shape=shape),

            tf.keras.layers.MaxPooling1D(),

            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=3,
                                   activation='relu'),

            tf.keras.layers.MaxPooling1D(),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.5, name='dropout2'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ])

    if (name == "recurrent1"):
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
            tf.keras.layers.Dense(10),
        ])

    if (name == "recurrent2"):
        strides = 2
        p_size = 2
        return tf.keras.Sequential([
            tf.keras.layers.Reshape((*shape, 1)),
            tf.keras.layers.MaxPooling2D((1, p_size), (1, strides)),
            tf.keras.layers.Reshape((shape[0], int(shape[1]/2))),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256)),
            tf.keras.layers.Dense(10),
        ])

    if (name == "convoulutional1"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3),
            tf.keras.layers.MaxPooling2D((2,2), (2,2)),
            tf.keras.layers.Conv2D(32, 3),
            tf.keras.layers.MaxPooling2D((2,2), (2,2)),
            tf.keras.layers.Conv2D(64, 3),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
            tf.keras.layers.Conv1D(128, 3),
            tf.keras.layers.MaxPooling2D((4, 4), (4, 4)),
            tf.keras.layers.Conv1D(64, 3),
            tf.keras.layers.MaxPooling2D((4, 4), (4, 4)),
        ])

    if (name == "PRCNN"):

        input = tf.keras.layers.Input(shape)
        r = tf.keras.layers.Reshape((*shape, 1))(input)
        r = tf.keras.layers.MaxPooling2D((1, 2), (1, 2))(r)
        r = tf.keras.layers.Reshape((shape[0], int(shape[1] / 2)))(r)
        r = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256))(r)

        c = tf.keras.layers.Conv2D(16, 3)(input)
        c = tf.keras.layers.MaxPooling2D((2,2), (2,2))(c)
        c = tf.keras.layers.Conv2D(32, 3)(c)
        c = tf.keras.layers.MaxPooling2D((2,2), (2,2))(c)
        c = tf.keras.layers.Conv2D(64, 3)(c)
        c = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(c)
        c = tf.keras.layers.Conv1D(128, 3)(c)
        c = tf.keras.layers.MaxPooling2D((4, 4), (4, 4))(c)
        c = tf.keras.layers.Conv1D(64, 3)(c)
        c = tf.keras.layers.MaxPooling2D((4, 4), (4, 4))(c)

        out = tf.keras.layers.Concatenate()[r, c]
        out = tf.keras.layers.Dense(10, activation="softmax")(out)

        return tf.keras.models.Model(inputs=input, outputs=out)


    # Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
    # Vanhoucke, V., Rabinovich, A., 2015. Going deeper with convolutions, in:
    # IEEE conference on computer vision and pattern recognition - można sprawdzić na temat jądra dla warstw splotowych
    # https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    if (name == "BBNN"):
        kernel_size = 3
        input = tf.keras.layers.Input(shape)
        i = tf.keras.layers.Reshape((*shape, 1))(input)
        c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=(1,1))(i)
        bn = tf.keras.layers.BatchNormalization()(c1)
        mp = tf.keras.layers.MaxPooling2D((4,1))(bn)

        shape = int(shape[0]/4), shape[1]
        inception_a = inception(shape, kernel_size, mp)
        connector1 = tf.keras.layers.Concatenate()([inception_a, mp])

        inception_b = inception(shape, kernel_size, connector1)
        connector2 = tf.keras.layers.Concatenate()([connector1, inception_b])

        inception_c = inception(shape, kernel_size, connector2)
        connector3 = tf.keras.layers.Concatenate()([connector2, inception_c])

        out = tf.keras.layers.BatchNormalization()(connector3)
        out = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1,1))(out)
        out = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.keras.layers.Dense(10, activation="softmax")(out)
        out = tf.keras.layers.Softmax()(out)

        return tf.keras.models.Model(inputs=input, outputs=out)


def inception(shape, kernel_size, _input):

    input = _input

    _1 = tf.keras.layers.BatchNormalization()(input)
    _1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(_1)

    _2 = tf.keras.layers.BatchNormalization()(input)
    _2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(_2)
    _2 = tf.keras.layers.BatchNormalization()(_2)
    _2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(_2)

    _3 = tf.keras.layers.BatchNormalization()(input)
    _3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(_3)
    _3 = tf.keras.layers.BatchNormalization()(_3)
    _3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding="same")(_3)

    _4 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=(1, 1), padding="same")(input)
    _4 = tf.keras.layers.BatchNormalization()(_4)
    _4 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(_4)

    return tf.keras.layers.Concatenate(axis=-1)([_1, _2, _3, _4])
