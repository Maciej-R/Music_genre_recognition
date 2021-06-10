from os import path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from settings import *
from data_processing import *
from models import make_model

model_name = "recurrent2"

split = 80  # % of data used for training (rest for validation)
filename = path.join(example_path, "all")
raw_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=1)

features = []
labels = []
lengths = []
for r in raw_dataset:
    rr = dset_parser(r)
    features.append(rr[0])
    labels.append(rr[1])
    lengths.append(len(rr[0]))

idxs = dict()
for i in range(len(genres)):
    idxs[genres[i]] = i
# Wyłoży się jak będą różnej długości
m = min(lengths)
for i in range(len(features)):
    if model_name == "recurrent1":  # In order to limit number of possible values for embedding
        s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
        s = s / (2**5)
        s = s.astype(np.uint16)
    else:
        if transpose:
            s = np.transpose(spectrogram(np.array(features[i][0:m]).astype(np.float32)))
        else:
            s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
    features[i] = tf.constant(np.reshape(s, (1, *s.shape)))
    labels[i] = tf.reshape(tf.constant(tf.one_hot(idxs[labels[i]], len(genres))), (1, len(genres)))

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#dataset.map(lambda data, label: (spectrogram(data), label))  # Looks like shallow copy issues
dataset = dataset.shuffle(min(len(dataset), 4000))
n_split = np.ceil(len(dataset) * split / 100)
training_dset = dataset.take(n_split)
validation_dset = dataset.skip(n_split)

_shape = features[0].shape[1:]
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

model = make_model(_shape, model_name)

#model.summary()
#tf.keras.utils.plot_model(model, model_name+".png")  # Requires graphviz installed (in system)
#exit(0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

history = model.fit(
    training_dset,
    epochs=1,
    validation_data=validation_dset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

labels_original = np.concatenate([y for x, y in validation_dset], axis=0)
labels_numeric = list()
for l in labels_original:
    idx = np.where(l == 1)[0][0]
    labels_numeric.append(idx)
predictions = model.predict(np.concatenate([x for x, y in validation_dset], axis=0))
predictions_numeric = list()
for p in predictions:
    idx = np.where(p == max(p))[0][0]
    predictions_numeric.append(idx)

cm = tf.math.confusion_matrix(labels_numeric, predictions_numeric)
print(cm)
print(genres)

#https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
exit(0)
