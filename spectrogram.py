# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Imports & File functions

import torch
import torchaudio
import matplotlib.pyplot as plt
import os
folders = ["data", "data/recordings", "data/subset", "img/spectograms"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_files(path):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(path) if isfile(join(path, f))]


# ### Save the spectrogram to a .png file

def save_spectrogram(data, filename):
    height = 200
    width = height*2

    fig.set_size_inches(width, height)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(2000)
    fig.add_axes(ax)
    ax.imshow(data, cmap='gray', aspect='auto')
    
    fig.savefig("img/spectograms/" + filename + ".png", dpi=1)
    fig.clf()


# ### Tell it where to find the modified files

# params to find the files
modified_filenames = get_files("audio_files/modified")
original_filenames = get_files("audio_files/original")

# ### Find length of the shortest audio file

# The result of this function should ALWAYS be higher than the value of ```ax.set_xlim()``` in ```save_spectrogram()```!

# +
lengths = []

print("Determining size of the shortest audio file")
total_number_files = len(modified_filenames) + len(original_filenames)
for index, file in enumerate(modified_filenames):
    # load file and make spectrogram
    waveform, sample_rate = torchaudio.load("audio_files/modified/" + file)
    specgram = torchaudio.transforms.MelSpectrogram(n_fft=4000, win_length=400)(waveform)
    lengths.append(specgram.size()[2])
    print(f"Progress:\t{round(((1+index)/total_number_files)*100, 2)}%", end='\r')

for index, file in enumerate(original_filenames):
    # load file and make spectrogram
    waveform, sample_rate = torchaudio.load("audio_files/original/" + file)
    specgram = torchaudio.transforms.MelSpectrogram(n_fft=4000, win_length=400)(waveform)
    lengths.append(specgram.size()[2])
    print(f"Progress:\t{round(((1+len(modified_filenames)+index)/total_number_files)*100, 2)}%", end='\r')

min_length = min(lengths)
print(f"Size of shortest audio file:\t{min_length}")
# -

# ### Generate Spectrograms

# +
#initialize the figure once so we dont get memory issues
fig = plt.figure(frameon=False)

# The spectograms for the mofied audio samples and original samples is stored in seperate 
# directories. This is done such that the ratio between modified and original samples can be
# controlled
print("Creating spectograms")
for index, file in enumerate(modified_filenames):
    # load file and make spectrogram
    waveform, sample_rate = torchaudio.load("audio_files/modified/" + file)
    specgram = torchaudio.compliance.kaldi.spectrogram(waveform, dither=0.)
    
    # plot spectrogram and store it
    save_spectrogram(specgram.t().numpy(), file[:-4])

    print(f"Progress:\t{round(((1+index)/total_number_files)*100, 2)}%", end='\r')
    
for index, file in enumerate(original_filenames):
    # load file and make spectrogram
    waveform, sample_rate = torchaudio.load("audio_files/original/" + file)
    specgram = torchaudio.compliance.kaldi.spectrogram(waveform, dither=0.)
    
    # plot spectrogram and store it
    save_spectrogram(specgram.t().numpy(), file[:-4])

    print(f"Progress:\t{round(((1+len(modified_filenames)+index)/total_number_files)*100, 2)}%", end='\r')

print("Finished making spectograms")
# -


