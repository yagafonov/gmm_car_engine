import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from librosa.util import buf_to_float
from librosa.core import stft


def cut_wav(path_to_wav, start_time, end_time):
    _, wav_data = wavfile.read(path_to_wav)

    return wav_data[int(sr * start_time): int(sr * end_time)]


def get_stft(wav_data):
    feat = np.abs(stft(buf_to_float(wav_data), n_fft=fft_size, hop_length=fft_step))

    return feat.T


# параметры для рассчета спектрограммы
fft_size = 512
fft_step = 128
sr = 16000

# временная разметка для обучения
time_list = [
    [3.6, 4.05],
    [9.14, 21.],
    [22.5, 31.6],
    [40.5, 49.7],
    [51.2, 66.],
]

# время звучания поломок
broken_times = [
    [2.3, 3.6],
    [4.5, 8.67],
    [20.59, 21.97],
    [32., 40.2],
    [66.5, 73.0],
    [74.5, 75.5],
    [49.5, 50.5],
]

if __name__ == '__main__':
    wav_path = './engine_sound.wav'
    train_features = []
    # готовим признаки для обучения, time_list - содержит разметку данных
    for [ts, te] in time_list:
        wav_part = cut_wav(wav_path, ts, te)
        spec = get_stft(wav_part)
        train_features.append(spec)
    X_train = np.vstack(train_features)

    # готовим признаки для теста
    full_wav_data = wavfile.read(wav_path)[1]
    X_test = get_stft(full_wav_data)

    gmm_clf = GaussianMixture(n_components=3)
    gmm_clf.fit(X_train)

    n_seconds = len(full_wav_data) // sr
    gmm_scores = []
    # правдоподобие на каждую секунду
    for i in range(n_seconds - 1):
        test_sec = X_test[(i * sr) // fft_step: ((i + 1) * sr) // fft_step, :]
        sc = gmm_clf.score(test_sec)
        gmm_scores.append(sc)

    # отображние спектрограммы
    plt.figure(figsize=(13, 8))
    plt.subplot(2, 1, 1)
    spec, _, _, _ = plt.specgram(full_wav_data, fft_size, sr, fft_size - fft_step, scale='dB', cmap='GnBu')
    for [s, e] in broken_times:
        plt.axvline(x=s, color='magenta')
        plt.axvline(x=e, color='red')
    plt.ylabel('Частота, Гц')
    plt.title('Спектрограмма')

    # отображение правдоподобия
    plt.subplot(2, 1, 2)
    plt.plot(list(range(n_seconds - 1)), gmm_scores)
    add_legend = True
    for [s, e] in broken_times:
        if add_legend:
            plt.axvline(x=s, color='magenta', label='Начало поломки')
            plt.axvline(x=e, color='red', label='Конец поломки')
            add_legend = False
        else:
            plt.axvline(x=s, color='magenta')
            plt.axvline(x=e, color='red')
    plt.axhline(y=10, color='green', label='Порог')

    plt.grid()
    plt.xlabel('Время, сек')
    plt.ylabel('Правдоподобие')
    plt.xlim((0, n_seconds - 1))
    plt.legend()

    plt.show()
