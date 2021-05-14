import scipy as sp
from os import path
from random import randint
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.colors as colors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from settings import *
from functools import partial


# Editable parameters
n_fft = 1024
hop_length = 256
n_mels = 256
window_attenutaion = 50
show_spectra = 0
use_log_scale = 0  # Spectrogram in dB if 1 [X = 10*log10(X)], default should be 0
use_mel_scale = 1  # Converting spectrogram to mel scale, default should be 1
s_power = 1  # 1 for energy, 2 for power [output data = FFT(music)^s_power]

window = sp.signal.windows.chebwin(n_fft, window_attenutaion, True)
filt = librosa.filters.mel(fs, n_fft, n_mels=n_mels)

testing = 0
BATCH_SIZE = 10
split = 80  # % of data used for training (rest for validation)


def spectrogram(data):
    """
    Uses window and mel filter from global settings
    :param data: Input signal
    :return: Spectrogram in mel scale of input data
    """
    spectrum = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window=window)
    spectrum = np.abs(spectrum).astype(dtype=np.float32)
    spectrum **= s_power
    if use_log_scale:  # Log scale
        spectrum = 10 * np.log10(spectrum)
    if use_mel_scale:  # Mel scale
        spectrum = filt.dot(spectrum)
    return spectrum


def dset_parser(raw_record):
    description = {
        "label": tf.io.VarLenFeature(tf.string),
        "data": tf.io.VarLenFeature(tf.int64)
    }
    try:
        content = tf.io.parse_single_example(raw_record, description)
    except Exception as e:
        print(str(e)[0:300] + str(e)[-200:-1])
    d = content['data'].values.numpy()  # Getting data values
    l = content['label'].values.numpy()[0].decode("utf-8")  # Getting label

    return (d, l)


def make_model(shape):
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
        tf.keras.layers.Dense(10, name='dense3')
    ])


# Displaying spectrograms with requested parameters of random song from each category from GTZAN
if show_spectra:
    fig, ax = plt.subplots(1, len(genres))
    for i in range(len(genres)):
        g = genres[i]
        N = randint(0, 99)
        if len(str(N)) < 2:
            rec = "0" * 4 + str(N)
        else:
            rec = "0" * 3 + str(N)
        _path = path.join(music_path, g)
        _path = path.join(_path, g + "." + rec + ".wav")
        try:
            fs_read, data = wavfile.read(_path)
        except Exception as e:
            print("Could not load music file: " + e)
        if fs_read != fs:
            raise RuntimeError()
        data = data.astype(dtype=np.float32)
        sgram = spectrogram(data)

        im = ax[i].imshow(sgram.transpose(), cmap=plt.get_cmap("Greens"),  # Greens, plasma, autum, hot
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=np.min(sgram), vmax=np.max(sgram), base=10))
        ax[i].set_title(g)
        ax[i].invert_yaxis()
        ticks_locations = np.arange(0, np.max(sgram.shape), 200)
        ax[i].set_yticks(ticks_locations)
        ax[i].set_yticklabels(np.round(ticks_locations * hop_length / fs, 2))
        if use_mel_scale:
            ax[i].set_xticks(np.arange(0, n_mels, 128))
        else:
            rng = np.arange(0, n_fft/2, 100)
            ax[i].set_xticks(np.round(rng))
            ax[i].set_xticklabels(np.round(rng / n_fft * fs), rotation="vertical")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("FFT in mel scale values", rotation=-90, va="bottom")
    fig.text(0.44, 0.04, 'Mels (plot uses SymLogNorm regardless of data transforms)', ha='center', va='center')

    plt.show()

# End show_spectra

filename = path.join(example_path, "all")
raw_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=1)

if testing:
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        dset_parser(raw_record)
        description = {
            "label": tf.io.VarLenFeature(tf.string),
            "data": tf.io.VarLenFeature(tf.int64)
        }
        try:
            content = tf.io.parse_single_example(raw_record, description)
        except Exception as e:
            print(str(e)[0:300] + str(e)[-200:-1])
        content['data'].values.numpy()  # Getting data values
        content['label'].values.numpy()[0].decode("utf-8")  # Getting label

features = []
labels = []
lengths = []
for r in raw_dataset:
    rr = dset_parser(r)
    features.append(rr[0])
    labels.append(rr[1])
    lengths.append(len(rr[0]))

# Wyłoży się jak będą różnej długości
m = min(lengths)
for i in range(len(features)):
    s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
    features[i] = tf.constant(np.reshape(s, (1, *s.shape)))

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#dataset.map(lambda data, label: (spectrogram(data), label))
#for d in dataset:
#    print(d)
dataset.shuffle(min(len(dataset), 4000))
n_split = np.ceil(len(dataset) * split / 100)
training_dset = dataset.take(n_split)
validation_dset = dataset.skip(n_split)

_shape = n_mels, int(np.ceil(m / hop_length + 1))
training_dset.batch(BATCH_SIZE)
validation_dset.batch(BATCH_SIZE)

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "genres.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

model = make_model(_shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    training_dset,
    epochs=2,
    validation_data=validation_dset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

#https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
exit(0)

with strategy.scope():
  # create the model


  #compile
  model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

 # train the model
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
EPOCHS = 100
raw_audio_history = model.fit(training_dataset_1d, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset_1d, epochs=EPOCHS,
                    callbacks=tensorboard_callback)

# evaluate on the test data
model.evaluate(testing_dataset_1d)

#############################################################

self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
self.gru = tf.keras.layers.GRU(rnn_units,
                               return_sequences=True,
                               return_state=True)
self.dense = tf.keras.layers.Dense(vocab_size)