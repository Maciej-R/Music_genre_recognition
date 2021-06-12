from settings import *
from data_processing import spectrogram, dset_parser
from os import path
import tensorflow as tf
import numpy as np
from scipy.io import wavfile


def read(model_name, n):
    """:arg n Number of files to read per genre - not applicable for serialized data"""

    g_len = len(genres)
    idxs = dict()
    for i in range(len(genres)):
        idxs[genres[i]] = i

    if read_serialized:

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

        m = min(lengths)  # All songs needs to be the same length
        for i in range(len(features)):
            if model_name == "recurrent1":  # In order to limit number of possible values for embedding
                s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
                s = s / (2 ** 5)
                s = s.astype(np.uint16)
            else:
                if transpose:
                    s = np.transpose(spectrogram(np.array(features[i][0:m]).astype(np.float32)))
                else:
                    s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
            features[i] = tf.constant(np.reshape(s, (1, *s.shape)))
            labels[i] = tf.reshape(tf.constant(tf.one_hot(idxs[labels[i]], g_len)), (1, g_len))

        _shape = features[0].shape[1:]
        return tf.data.Dataset.from_tensor_slices((features, labels)), _shape

    else:
        features = list()
        labels = list()
        m = float("inf")
        for genre in genres:
            g_idx = idxs[genre]
            path_main = path.join(music_path, genre)
            for i in range(n):
                _path = path_main
                if i < 10:
                    filename = "0" * 4 + str(i)
                elif i < 100:
                    filename = "0" * 3 + str(i)
                else:
                    filename = "0" * 2 + str(i)
                _path = path.join(_path, genre + "." + filename + ".wav")
                try:
                    fs_read, data = wavfile.read(_path)
                    tmp = len(data)
                    if tmp < m:
                        m = tmp
                except Exception as e:
                    print(e)
                    continue
                assert(fs_read == fs)

                features.append(data)
                labels.append(tf.reshape(tf.one_hot(g_idx, g_len), (1, g_len)))

        for i in range(len(features)):
            s = spectrogram(np.array(features[i][0:m]).astype(np.float32))
            features[i] = np.reshape(s, (1, *s.shape))
            if transpose:
                features[i] = np.transpose(features[i])

        _shape = features[0].shape[1:]
        return tf.data.Dataset.from_tensor_slices((features, labels)), _shape
