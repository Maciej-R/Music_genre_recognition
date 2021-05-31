# Not editable parameters
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
fs = 22050
# Editable parameters
n_fft = 1024
hop_length = 256
n_mels = 256
window_attenutaion = 50
show_spectra = 0
use_log_scale = 0  # Spectrogram in dB if 1 [X = 10*log10(X)], default should be 0
use_mel_scale = 1  # Converting spectrogram to mel scale, default should be 1
s_power = 1  # 1 for energy, 2 for power [output data = FFT(music)^s_power]
# Paths
data_path = "C:\\Users\\JaroslawZelechowski\\Documents\\Music_genre_recognition\\Data"
music_path = data_path + "\\genres_original"
example_path = data_path + "\\examples"

# Models
save_model = 1
model_path = data_path + "\\models\\"

BATCH_SIZE = 10