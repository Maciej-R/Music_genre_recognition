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
            tf.keras.layers.Reshape((*shape, 1)),
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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

    if (name =="conv_zporadnika"):
        X_input = tf.keras.layers.Input(shape)

        X = tf.keras.layers.Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((2,2))(X)
        
        X = tf.keras.layers.Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((2,2))(X)
        
        X = tf.keras.layers.Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((2,2))(X)

        X = tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((2,2))(X)
        
        X = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((2,2))(X)

        
        X = tf.keras.layers.Flatten()(X)
        
        X = tf.keras.layers.Dropout(rate=0.3)

        X = tf.keras.layers.Dense(10, activation='softmax', name='fc' + str(10))(X)

        return tf.keras.layers.Model(inputs=X_input,outputs=X,name='GenreModel')

    if (name == "PRCNN"):
        input = tf.keras.layers.Input(shape)
        inn = tf.keras.layers.Reshape((*shape, 1))(input)

        r = tf.keras.layers.MaxPooling2D((1, 2), (1, 2))(inn)
        r = tf.keras.layers.Reshape((shape[0], int(shape[1] / 2)))(r)  # MaxPooling drops half
        r = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256))(r)
        #r = tf.keras.layers.Dense(5, activation="softmax")(r)

        n_filters = 16
        c = tf.keras.layers.Conv2D(n_filters, 3)(inn)
        c = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(c)
        c = tf.keras.layers.Conv2D(n_filters * 2, 3)(c)
        c = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(c)
        c = tf.keras.layers.Conv2D(n_filters * 4, 3)(c)
        c = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(c)
        c = tf.keras.layers.Conv2D(n_filters * 8, 3)(c)
        c = tf.keras.layers.MaxPooling2D((4, 4), (4, 4))(c)
        c = tf.keras.layers.Conv2D(n_filters * 4, 3)(c)
        c = tf.keras.layers.MaxPooling2D((4, 4), (4, 4))(c)
        c = tf.keras.layers.Flatten()(c)
        #c = tf.keras.layers.Dense(5, activation="softmax")(c)

        out = tf.keras.layers.Concatenate(axis=1)([r, c])
        out = tf.keras.layers.Dense(10)(out)

        return tf.keras.models.Model(inputs=input, outputs=out)


    # Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
    # Vanhoucke, V., Rabinovich, A., 2015. Going deeper with convolutions, in:
    # IEEE conference on computer vision and pattern recognition - można sprawdzić na temat jądra dla warstw splotowych
    # https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    if (name == "BBNN"):

        kernel_size = 3
        input, mp = BBNN_initial_layers(shape, kernel_size)

        shape = int(shape[0]/4), shape[1]
        inception_a = inception(shape, kernel_size, mp)
        connector1 = tf.keras.layers.Concatenate(axis=3)([inception_a, mp])

        inception_b = inception(shape, kernel_size, connector1)
        connector2 = tf.keras.layers.Concatenate(axis=3)([connector1, inception_b])

        inception_c = inception(shape, kernel_size, connector2)
        connector3 = tf.keras.layers.Concatenate(axis=3)([connector2, inception_c])

        out = BBNN_final_layers(connector3)

        return tf.keras.models.Model(inputs=input, outputs=out)

    if (name == "BBNN_simplified"):

        kernel_size = 3

        input, mp = BBNN_initial_layers(shape, kernel_size)

        shape = int(shape[0]/4), shape[1]
        inception_a = inception(shape, kernel_size, mp)
        #connector1 = tf.keras.layers.Concatenate(axis=3)([inception_a, mp])

        out = BBNN_final_layers(inception_a)

        return tf.keras.models.Model(inputs=input, outputs=out)



def inception(shape, kernel_size, _input):

    input = _input

    #_1 = tf.keras.layers.BatchNormalization()(input)
    _1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(input)

    #_2 = tf.keras.layers.BatchNormalization()(input)
    _2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(input)
    #_2 = tf.keras.layers.BatchNormalization()(_2)
    _2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(_2)

    #_3 = tf.keras.layers.BatchNormalization()(input)
    _3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(input)
    #_3 = tf.keras.layers.BatchNormalization()(_3)
    _3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding="same")(_3)

    _4 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=(1, 1), padding="same")(input)
    #_4 = tf.keras.layers.BatchNormalization()(_4)
    _4 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding="same")(_4)

    return tf.keras.layers.Concatenate(axis=-1)([_1, _2, _3, _4])


def BBNN_final_layers(_input):

    #out = tf.keras.layers.BatchNormalization()(_input)
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1))(_input)
    out = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(out)
    #out = tf.keras.layers.BatchNormalization()(out)
    #out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(10)(out)
    return tf.keras.layers.Softmax()(out)


def BBNN_initial_layers(shape, kernel_size):

    input = tf.keras.layers.Input(shape)
    i = tf.keras.layers.Reshape((*shape, 1))(input)
    c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=(1, 1))(i)
    bn = tf.keras.layers.BatchNormalization()(c1)
    return input, tf.keras.layers.MaxPooling2D((4, 1))(bn)