# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import os

df = pd.read_csv ('speakers_all.csv')
df

# +
# Given the Histogram, these are the native languages with at least 15 audio recordings (speakers?).

names = ['english','spanish','arabic','mandarin','french','korean','portuguese','russian','dutch','turkish','german',
         'polish','italian','japanese','macedonian','cantonese','farsi','vietnamese','romanian','swedish','amharic',
         'tagalog','bulgarian','hindi','serbian','bengali','urdu','greek','thai']

countries = ['usa','south korea','china']
native_languages = ['english','korean','mandarin']

column_names = df.columns
# Creation of the final dataframe with only the desired files.
final_df= pd.DataFrame(columns = column_names)

path = os.getcwd()
directory_names = []

# We create directories to store the different recordings based on the native language
for i, name in enumerate(countries):
    directory_names.append(path + "/" + name + native_languages[i])

    try:
        os.mkdir(directory_names[i])
    except OSError:
        print ("Creation of the directory %s failed" % directory_names[i])
    else:
        print ("Successfully created the directory %s " % directory_names[i])
        
    

# +
import shutil 

count=0
recordings_path=path+'/recordings/recordings'

# We are saving the audio recordings and data of the desired native languages in their corresponding
    # folders to have a cleaner structure.
for index, native_lan in enumerate(df['native_language']):
    try:
        i = native_languages.index(native_lan)
        if df['country'][index] == countries[i]:
            final_df.loc[df.index[count]] = df.iloc[index]
            filename = final_df['filename'][count]
            source = recordings_path + "/" + filename + '.mp3'
            destination = path + "/" + countries[i] + native_lan + "/"+ filename+'.mp3'
            shutil.copyfile(source, destination) 
            count+=1
    except:
        continue
# -

# Exporting the csv.
final_df.to_csv('final_csv.csv')

# +
import librosa
from pydub import AudioSegment

file_name = path+'\\german2.wav'

y, sr = librosa.load(file_name)


# +
librosa.feature.melspectrogram(y=y, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

import matplotlib.pyplot as plt
import numpy as np
import librosa.display

plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
# -

native_languages = ['english','korean','mandarin']
try:
    
    index = native_languages.index('english')
    print(index)
except:
    index = -1

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
print("grey scale")
librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear power spectrogram (grayscale)')


