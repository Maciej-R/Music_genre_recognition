import scipy as sp
from os import path
from random import randint
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.colors as colors

data_path = "D:\\Data\\archive\\Data"
music_path = "D:\\Data\\archive\\Data\\genres_original"
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "jazz", "metal", "pop", "reggae", "rock"]
fs = 22050
n_fft = 1024
hop_length = 256
n_mels = 256
window_attenutaion = 50
window = sp.signal.windows.chebwin(n_fft, window_attenutaion, True)

show_spectra = 1

if show_spectra:
    filt = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
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
        fs_read, data = wavfile.read(_path)
        if fs_read != fs:
            raise RuntimeError()
        data = data.astype(dtype=np.float32)
        energy = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window=window)
        energy = np.abs(energy).astype(dtype=np.float32)
        #power = librosa.stft(data, fs, n_fft=n_fft, hop_length=hop_length, power=2, window=window)
        mel = energy
        mel = filt.dot(mel)

        im = ax[i].imshow(mel.transpose(), cmap=plt.get_cmap("Greens"),  #Greens, plasma, autum, hot
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=np.min(mel), vmax=np.max(mel), base=10))
        ax[i].set_title(g)
        ax[i].invert_yaxis()
        ticks_locations = np.arange(0, np.max(mel.shape), 200)
        ax[i].set_yticks(ticks_locations)
        ax[i].set_yticklabels(np.round(ticks_locations * hop_length / fs, 2))
        ax[i].set_xticks(np.arange(0, n_mels, 128))

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("FFT in mel scale values", rotation=-90, va="bottom")
    fig.text(0.44, 0.04, 'Mels', ha='center', va='center')

    plt.show()
