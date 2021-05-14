import tensorflow as tf
from scipy.io import wavfile
from settings import *
from os import path, makedirs, remove


if not path.exists(example_path):
    inp = input("Do you want to create dir " + example_path + "? [y]")
    if inp == "y":
        makedirs(example_path)
    else:
        print("Specify path in settings.py for tf.Example to be saved")
        raise RuntimeError()

try:
    p = path.join(example_path, "all")
    if path.exists(p):
        remove(p)
    with tf.io.TFRecordWriter(p) as file_writer:
        for genre in genres:
            examples = list()
            label = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(genre, "utf-8")]))
            path_main = path.join(music_path, genre)
            for i in range(2):
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
                except Exception as e:
                    print(e)
                    continue
                assert(fs_read == fs)

                feature = tf.train.Feature(int64_list=tf.train.Int64List(value=data))
                examples.append(tf.train.Example(features=tf.train.Features(feature={"label":label, "data":feature})))

            # Write the records to a file.
            for ex in examples:
                ex = ex.SerializeToString()
                file_writer.write(ex)

except Exception as e:
    print("Error writing file")
    print(str(e)[0:300])

