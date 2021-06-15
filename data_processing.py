import librosa
import scipy as sp
import numpy as np
from os import path
import tensorflow as tf
from settings import *
from random import randint
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.colors as colors

window = sp.signal.windows.chebwin(n_fft, window_attenutaion, True)
filt = librosa.filters.mel(fs, n_fft, n_mels=n_mels)


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
    if normalize_spectrogram:
        s = np.std(spectrum)
        a = np.average(spectrum)
        spectrum -= a
        spectrum /= s
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