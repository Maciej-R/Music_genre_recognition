import scipy as sp
from os import path
from random import randint
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.colors as colors


# Not editable parameters
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "jazz", "metal", "pop", "reggae", "rock"]
fs = 22050
# Paths
data_path = "D:\\Data\\archive\\Data"
music_path = "D:\\Data\\archive\\Data\\genres_original"
# Editable parameters
n_fft = 1024
hop_length = 256
n_mels = 256
window_attenutaion = 50
show_spectra = 1
use_log_scale = 0  # Spectrogram in dB if 1 [X = 10*log10(X)], default should be 0
use_mel_scale = 0  # Converting spectrogram to mel scale, default should be 1
s_power = 1  # 1 for energy, 2 for power [output data = FFT(music)^s_power]

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
    return spectrum


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
