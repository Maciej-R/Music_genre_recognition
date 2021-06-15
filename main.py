#from os import putenv
#putenv("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from settings import *
from data_processing import *
from models import make_model
from data_reading import read


model_name = "convoulutional1"

EPOCHS = 15
# MODELS = ["test", "recurrent1", "recurrent2", "convoulutional1", "conv_zporadnika", "PRCNN", "BBNN", "BBNN_simplified"]
# print('Wybierz model:')
# for i in range(len(MODELS)):
#     print(f'{i+1}. {MODELS[i]}')
# model_name = MODELS[int(input())-1]

# split = 80  # % of data used for training (rest for validation)

dataset, _shape = read(model_name, 20)
#dataset.map(lambda data, label: (spectrogram(data), label))  # Looks like shallow copy issues
dataset = dataset.shuffle(min(len(dataset), 4000))
n_split = np.ceil(len(dataset) * 0.7)  # 70% for training
training_dset = dataset.take(n_split)
validation_dset = dataset.skip(n_split)
n_split = np.ceil(len(dataset) * 0.2)  # 20% for validation, 10% for testing
test_dset = validation_dset.skip(n_split)
validation_dset = validation_dset.take(n_split)

training_dset.batch(BATCH_SIZE)
validation_dset.batch(BATCH_SIZE)
test_dset.batch(BATCH_SIZE)

learn = True
if learn:

    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"checkpoints/{model_name}.h5", save_best_only=True, monitor='val_accuracy'
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor='val_accuracy'
    )

    model = make_model(_shape, model_name)

    #tf.keras.utils.plot_model(model, model_name+".png")  # Requires graphviz installed (in system)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(), 'accuracy'],
        run_eagerly=True
    )

    history = model.fit(
        training_dset,
        epochs=EPOCHS,
        validation_data=validation_dset,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

else:
    model = tf.keras.models.load_model(f"checkpoints/{model_name}.h5")


model.evaluate(test_dset, steps=test_dset.__len__().numpy()/10, batch_size=10)

labels_original = np.concatenate([y for x, y in test_dset], axis=0)
labels_numeric = list()
for l in labels_original:
    idx = np.where(l == 1)[0][0]
    labels_numeric.append(idx)

predictions = model.predict(np.concatenate([x for x, y in test_dset], axis=0), batch_size=BATCH_SIZE)

#predictions = model.apply(np.concatenate([x for x, y in test_dset], axis=0))
#predictions = model(np.concatenate([x for x, y in test_dset], axis=0))
#model.evaluate(np.concatenate([x for x, y in test_dset], axis=0), steps=test_dset.__len__().numpy()/10, batch_size=10)
# model.evaluate(np.concatenate([x for x, y in validation_dset], axis=0), steps=validation_dset.__len__().numpy()/10, batch_size=10)

predictions_numeric = list()
for p in predictions:
    idx = np.argmax(p)
    predictions_numeric.append(idx)
counter = 0

for i in range(len(labels_numeric)):
    if labels_numeric[i] == predictions_numeric[i]:
        counter += 1

print(counter/len(labels_numeric))

cm = tf.math.confusion_matrix(labels_numeric, predictions_numeric)

print(cm)
print(genres)

with open(f'confusion_matrixes/{model_name}_{counter/len(labels_numeric)}', 'w') as f:
    f.write(str(cm))

# print(tf.keras.metrics.binary_accuracy(labels_numeric, predictions_numeric))

con = tf.math.confusion_matrix(labels=labels_numeric, predictions=predictions_numeric )

#https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
exit(0)
