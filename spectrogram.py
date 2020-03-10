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

import os
folders = ["data", "img"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

import torch
import torchaudio
import matplotlib.pyplot as plt


def save_spectrogram(data, filename):
    height = 200
    #width = int((specgram.size()[2]/specgram.size()[1]) * height)
    width = height*2

    fig.set_size_inches(width, height)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(2000)
    fig.add_axes(ax)
    ax.imshow(data, cmap='gray', aspect='auto')
    
    fig.savefig("img/" + filename + ".png", dpi=1)
    fig.clf()


def get_files(path):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(path) if isfile(join(path, f))]


# +
#print("Shape of spectrogram: {}".format(specgram.size()))

data_path = "data/recordings"
all_filenames = get_files(data_path)
test_filenames = [x for x in all_filenames if "german" in x]
fig = plt.figure(frameon=False)

for file in test_filenames:
    waveform, sample_rate = torchaudio.load(data_path + "/" + file)
    
    specgram = torchaudio.transforms.MelSpectrogram(n_fft=4000, win_length=400)(waveform)
    save_spectrogram(specgram.log2()[0,:,:].numpy(), file[:-4] + "_mel")
    
#     specgram = torchaudio.compliance.kaldi.spectrogram(waveform, dither=0.)
#     save_spectrogram(specgram.t().numpy(), file[:-4] + "_kaldi")
# -


