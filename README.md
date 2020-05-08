
# Accent Classification
Using CNNs to classify the native tongue of a speaker

# General structure 
The project is designed in such a way that data-augmentation and the training/testing of the network can be done independant of each other. This way, it is easier to swap the data-set with another dataset of your liking. 

## Getting started
The required python packages are found in 'requirements.txt'. Please make sure these are installed before continuing. 
You can download the entire project by opening a terminal in the desired directory and either typing in this terminal the command:

```bash
git clone git@github.com:Skapke/AccentClassification.git
```

or the command:
```bash
git clone https://github.com/Skapke/AccentClassification.git
```

Move to the correct directory with the command:

```bash
cd AccentClassification
```

From now on, it is assumed that you are working from this directory

## Dataset
The original audio samples we used to train our network came from the [Speech Accent Archive](https://www.kaggle.com/rtatman/speech-accent-archive). From this dataset, we took the english, mandarin and korean samples and augmented these samples by increasing and decreasing the playback speed of these samples. The spectograms of all samples can be found in the '/img' folder. If you want to use these spectograms, you can jump the section *Network*. If you want to generate these spectograms yourself, you can download the dataset we used from google dive using [this link](https://drive.google.com/file/d/1zEh4pkqAB5rNDTOLA8g8YDI9p7uwXg2B/view?usp=sharingFILEID), or with the following command:

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zEh4pkqAB5rNDTOLA8g8YDI9p7uwXg2B' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zEh4pkqAB5rNDTOLA8g8YDI9p7uwXg2B" -O audio_files.tar && rm -rf /tmp/cookies.txt
```

And extract the files afterwards:
```bash
tar -xvf audio_files.tar
```
### Data augmentation
After downloading the audio files, create the spectograms using the 'spectogram.py' script:

```bash
python3 spectrogram.py
```
### Generating .csv files
In order have the correct labels accociated with the spectograms, each class has a .csv file in which the file name and the correct label are linked to each other. The original speech archive stored a lot more data then just the native language of the speaker. This implementation makes it possible to switch out the existing labels with a different label from the original .csv file.

To create the .csv files with labels for the current implementation, run:

```bash
python3 create_csv.py
```

## Network
Once all the data augmentation has finished, it is possible to start training and testing the different networks. To do this, one could use either the jupyter-notebook script 'LanClass.ipynb', or the python script directly by:
```bash
python3 LanClass.py
```


### Useful links
If interested in similar/related projects, you can check out the following links. 
#### PyTorch
* GitHub Project: [UrbanSound Classification](https://github.com/ksanjeevan/crnn-audio-classification)
* PyTorch Tutorial: [Spectograms with PyTorch](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html)
#### Keras
* GitHub Project: [Speech Accent Recognition](https://github.com/yatharthgarg/Speech-Accent-Recognition)
* Random Tutorial: [Urban Sound using Keras](https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4)
* Documentation: [Librosa](https://librosa.github.io/librosa/index.html)

