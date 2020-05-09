#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

n_epochs_Lenet = 10
n_epochs_Alexnet = 50
learning_rate = 0.03
momentum = 0.8
momentum_4 = 0.4
train_size = 80
test_size = 20

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# In[ ]:


## If using google colab, you'll need these two lines as well
#from google.colab import drive
#drive.mount('/content/gdrive')


# In[2]:


## First, we load the CSV Files that contais the information about the samples registered for each native language
english_data = pd.read_csv('img/english_subset.csv')
mandarin_data = pd.read_csv('img/mandarin_subset.csv')
korean_data = pd.read_csv('img/korean_subset.csv')


# In[3]:


## The objective of this class is to create items composed by a Tensor that represent and Image and a Tensor that represent a Label.
## The inputs will be a csv File with the information needed for each sample and the directory of the images and the CSV.

class LanClassDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        
        self.dataset = []
        self.labels = csv_file  

        for idx in range(len(self.labels)):
          img_name = os.path.join(root_dir,
                                self.labels['file_name'][idx])
          image = io.imread(img_name)
          transform = transforms.Compose([torchvision.transforms.ToPILImage(),
                                          torchvision.transforms.Resize([32,64]),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
          image = transform(image)
          self.dataset.append(image)

        self.dataset = torch.stack(self.dataset)
        self.labels,_ = pd.factorize(self.labels['native_language'])
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return np.size(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.dataset[idx]
        label = self.labels[idx]
        sample = {'image': image, 'label': label}
        return sample

    def __delitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        del self.dataset[idx]
        del self.labels[idx]


# In[4]:


## The objective of this function is to randomly select for each native language the training and the test
## samples. They will be added to the dataframes that will be used for the networks.

def Random_selection_samples():

  english_size = len(english_data)
  korean_size = len(korean_data)
  mandarin_size = len(mandarin_data)

  from random import randint

  english_vector = np.empty((0,1), int)
  english_train_vector = np.empty((0,1), int)
  english_test_vector = np.empty((0,1), int)
  korean_vector = np.empty((0,1), int)
  korean_train_vector = np.empty((0,1), int)
  korean_test_vector = np.empty((0,1), int)
  mandarin_vector = np.empty((0,1), int)
  mandarin_train_vector = np.empty((0,1), int)
  mandarin_test_vector = np.empty((0,1), int)

  ## all the indexes of the objects available
  for index in range (english_size):
    english_vector = np.append(english_vector,index)
  for index in range (korean_size):
    korean_vector = np.append(korean_vector,index)
  for index in range (mandarin_size):
    mandarin_vector = np.append(mandarin_vector,index)

  ## we randomly pick the train objects
  for index in range (train_size):
    value = randint(0, len(english_vector)-1)
    english_train_vector = np.append(english_train_vector,english_vector[value])
    english_vector = np.delete(english_vector,value)
  for index in range (train_size):
    value = randint(0, len(korean_vector)-1)
    korean_train_vector = np.append(korean_train_vector,korean_vector[value])
    korean_vector = np.delete(korean_vector,value)
  for index in range (train_size):
    value = randint(0, len(mandarin_vector)-1)
    mandarin_train_vector = np.append(mandarin_train_vector,mandarin_vector[value])
    mandarin_vector = np.delete(mandarin_vector,value)

  ## we randomly pick the test objects
  for index in range (test_size):
    value = randint(0, len(english_vector)-1)
    english_test_vector = np.append(english_test_vector,english_vector[value])
    english_vector = np.delete(english_vector,value)
  for index in range (test_size):
    value = randint(0, len(korean_vector)-1)
    korean_test_vector = np.append(korean_test_vector,korean_vector[value])
    korean_vector = np.delete(korean_vector,value)
  for index in range (test_size):
    value = randint(0, len(mandarin_vector)-1)
    mandarin_test_vector = np.append(mandarin_test_vector,mandarin_vector[value])
    mandarin_vector = np.delete(mandarin_vector,value)

  ## Now we add all the items to the dataframes
  column_names = ["file_name", "native_language"]

  df_train = pd.DataFrame(columns = column_names)
  df_test = pd.DataFrame(columns = column_names)

  ## First the test dataframe
  for value in (english_test_vector):
    df_test= df_test.append({'file_name' : english_data['file_name'][value] , 'native_language' : english_data['native_language'][value]} , ignore_index=True)
  for value in (korean_test_vector):
    df_test= df_test.append({'file_name' : korean_data['file_name'][value] , 'native_language' : korean_data['native_language'][value]} , ignore_index=True)
  for value in (mandarin_test_vector):
    df_test= df_test.append({'file_name' : mandarin_data['file_name'][value] , 'native_language' : mandarin_data['native_language'][value]} , ignore_index=True)

  ## Then the train dataframe
  for value in (english_train_vector):
    df_train= df_train.append({'file_name' : english_data['file_name'][value] , 'native_language' : english_data['native_language'][value]} , ignore_index=True)
  for value in (korean_train_vector):
    df_train= df_train.append({'file_name' : korean_data['file_name'][value] , 'native_language' : korean_data['native_language'][value]} , ignore_index=True)
  for value in (mandarin_train_vector):
    df_train= df_train.append({'file_name' : mandarin_data['file_name'][value] , 'native_language' : mandarin_data['native_language'][value]} , ignore_index=True)

  return df_train,df_test


# In[13]:


## This function has the goal of creating the items of the train and test loaders
def get_loaders():
  train_loader = LanClassDataset(csv_file=df_train,
                                      root_dir='img/spectograms/')
  test_loader = LanClassDataset(csv_file=df_test,
                                      root_dir='img/spectograms/')
  return train_loader, test_loader


# In[6]:


## This network is the adaptation of the LeNet architecture.

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 15, 2)
        self.fc1 = nn.Linear(15 * 7 * 15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 15 * 7 * 15)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[7]:


## This network is the adaptation of the AlexNet architecture.

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=2, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(28, 84, kernel_size=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(84, 168, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(168, 112, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(112, 112, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(112 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[8]:


## We initialize the network, the optimizer and the criterion.
network = AlexNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum = momentum)
criterion = nn.CrossEntropyLoss()


# In[9]:


def train(epoch):

  running_loss = 0.0
  for i,_ in enumerate(train_loader):
    # get the inputs; data is a list of [inputs, labels]
    images = train_loader.dataset
    labels = train_loader.labels

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = network(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # we print the loos obtained at the end of each train epoch
    running_loss += loss.item()
    if i == train_size*3-1:
      print('[%d, %5d] loss: %.3f' %
        (epoch, i + 1, running_loss / 20))
    running_loss = 0.0

  ## the state of the network is saved at the end of the train epoch
  torch.save(network.state_dict(), F"model.pth")


# In[10]:


def test():
  
  correct = 0
  total = 0
  with torch.no_grad():
    for i,_ in enumerate(test_loader):
        images = test_loader.dataset
        labels = test_loader.labels
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the test images: %.2f %%' % (
      100 * correct / total))


# In[11]:


def epoch(it_number):
  print('Epoch number %d:' % (it_number))
  train(it_number)
  test()


# In[15]:


## 5 simulations will be run, to select different random samples. Each of the selections
## will be tested for each of the Networks + Optimizers defined, to compare the 
## performance of each combination.

for simulation in range(1,6):
  print('Simulation number %d:' % (simulation))
  df_train,df_test = Random_selection_samples()
  print('Random samples taken')
  train_loader, test_loader = get_loaders()
  print('Data loaded')

  print('Network = Alexnet, Optimizer = SGD Mom = 0.8')
  network = AlexNet()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum = momentum)
  criterion = nn.CrossEntropyLoss() 
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Alexnet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

  print('Network = Lenet, Optimizer = Adam')
  network = LeNet()
  optimizer = optim.Adam(network.parameters(), lr=0.001,
                         betas=(0.9, 0.999),weight_decay = 0.002, 
                         amsgrad=False)
  criterion = nn.CrossEntropyLoss()
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Lenet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

  print('Network = Lenet, Optimizer = RMSProp')
  network = LeNet()
  optimizer = optim.RMSprop(network.parameters(), lr=0.001,
                          alpha=0.99)
  criterion = nn.CrossEntropyLoss()
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Lenet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

  print('Network = Lenet, Optimizer = SGD Mom = 0.8')
  network = LeNet()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum = momentum)
  criterion = nn.CrossEntropyLoss()
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Lenet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

  print('Network = Lenet, Optimizer = SGD Mom = 0.4')
  network = LeNet()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum = momentum_4)
  criterion = nn.CrossEntropyLoss()
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Lenet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

  print('Network = Lenet, Optimizer = SGD without momentum')
  network = LeNet()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()
  print(datetime.datetime.now().time())
  for epoch_number in range(1, n_epochs_Lenet + 1):
    epoch(epoch_number)
    print(datetime.datetime.now().time())

