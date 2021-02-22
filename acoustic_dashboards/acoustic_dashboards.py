import pyttsx3
from pydub import AudioSegment
from scipy import stats
from scipy.io import wavfile
import pandas as pd
import numpy as np
from functools import reduce
from operator import iadd

# from itertools import zip


def create_and_save_speech_sample(text: str, volume):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 175)
    engine.setProperty('volume', volume)
    # engine.say(str(text))
    engine.save_to_file(str(text), 'tmp/' + str(text) + '.mp3')
    engine.runAndWait()


def normalize_data(data_list):
    min_new = 0.25
    max_new = 1
    max_value = max(data_list)
    min_value = min(data_list)
    assert max_value > min_value, 'max_value must be bigger than min_value'
    norm_factor = (max_new - min_new) / (max_value - min_value)
    data_list_normed = [(i_value - min_value) * norm_factor + min_new
                        for i_value in data_list]
    return data_list_normed, norm_factor


def create_noise(length_in_seconds=3, amplitude=10):
    sample_rate = 44100  # CD Quality
    length_in_seconds = length_in_seconds
    amplitude = amplitude
    noise = stats.truncnorm(-1, 1, scale=2 ** amplitude).rvs(sample_rate * length_in_seconds)
    wavfile.write('tmp/noise.wav', sample_rate, noise.astype(np.int16))


def sample_outcome(x_data: list, y_data_normed: list):
    noise = AudioSegment.from_wav("tmp/noise.wav")

    def generate_mix_samples(x_data_sub, y_data_normed_sub):
        speech_sample = AudioSegment.from_wav("tmp/" + str(x_data_sub) + ".mp3")
        norm_factor = (y_data_normed_sub - min(y_data_normed))/(max(y_data_normed) - min(y_data_normed))
        noise_tmp = noise - 10*(1 - norm_factor)
        seconds = 2
        bias_for_speech = 5
        mixed = (noise_tmp[:seconds * 1E3] +
                 noise_tmp[seconds * 1E3 + 1:2 * seconds * 1E3].overlay(speech_sample + bias_for_speech * norm_factor) +
                 noise_tmp[seconds * 2 * 1E3:])
        return mixed

    mixed_list = [generate_mix_samples(x_data_sub=i_data[0], y_data_normed_sub=i_data[1])
                  for i_data in zip(x_data, y_data_normed)]
    final_mix = reduce(iadd, mixed_list)
    final_mix.export("tmp/final.mp3", format="mp3")
    return True
    # mixed.export("tmp/final_2.mp3", format="mp3")


class AcousticDashboards:
    df_data = pd.DataFrame

    def __init__(self):
        pass

    def load_data(self, file_name: str):
        if file_name[-3:] == 'csv':
            self.df_data = pd.read_csv('data/' + file_name, sep=';', decimal=',')

    def generate_discrete_1d_chart(self, x_column_name, y_column_name):
        x_data = self.df_data[x_column_name].values
        y_data = self.df_data[y_column_name].values
        y_data_normed, norm_factor = normalize_data(y_data)

        [create_and_save_speech_sample(text=i_data[0], volume=i_data[1]) for i_data in zip(x_data, y_data_normed)]
        sample_outcome(x_data=x_data, y_data_normed=y_data_normed)