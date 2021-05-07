import tensorflow as tf
from scipy.io import wavfile
from settings import *
from os import path, makedirs, remove
import numpy as np


if not path.exists(example_path):
    inp = input("Do you want to create dir " + example_path + "? [y]")
    if inp == "y":
        makedirs(example_path)
    else:
        print("Specify path in settings.py for tf.Example to be saved")
        raise RuntimeError()

for genre in genres:
    feature = dict()
    _path = path.join(music_path, genre)
    for i in range(1):
        if i < 10:
            filename = "0" * 4 + str(i)
        elif i < 100:
            filename = "0" * 3 + str(i)
        else:
            filename = "0" * 2 + str(i)
        _path = path.join(_path, genre + "." + filename + ".wav")
        try:
            fs_read, data = wavfile.read(_path)
        except Exception as e:
            print(e)
            continue
        assert(fs_read == fs)

        feature[filename] = tf.train.Feature(int64_list=tf.train.Int64List(value=data))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example.SerializeToString()

    # Write the records to a file.
    try:
        p = path.join(example_path, genre)
        if path.exists(p):
            remove(p)
        with tf.io.TFRecordWriter(p) as file_writer:
            file_writer.write(example)
    except Exception as e:
        print("Error writing file")
        print(str(e)[0:200])

filenames = []
for g in genres:
    filenames.append(path.join(example_path, g))

raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    s = str(example)
    print(s[0:100])
    break
