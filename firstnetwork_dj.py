# ##### Imports

# +
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


# -

# ##### Helper functions

# Function to display an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ### Load data

# +
custom_transform = transforms.Compose([transforms.Grayscale(1),
                                       transforms.Resize((32,32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                      ])
    
def load_trainset():
    train_path = 'img/dj/train/'
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=custom_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    return train_loader

def load_testset():
    test_path = 'img/dj/test/'
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=custom_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)
    return test_loader

# Batch size is set to 1 because we do not have a lot of data


# -

# ### View a random training image

# +
# get some random training image
for train_images, train_labels in load_trainset():
    random_int = random.randint(0, len(train_images)-1)
    sample_image = train_images[random_int]
    sample_label = train_labels[random_int]

# show images
imshow(torchvision.utils.make_grid(sample_image))
print(sample_label)
# -

# ### Define network

# CIFAR-10 Network: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Neural Networks tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

# +
# 0 is CIFAR-10
# 1 is NN
network_type = 0

# CIFAR-10
if network_type == 0:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5) #in, out, kernel size
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    net = Net()
    
# NN
if network_type == 1:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = Net()
# -

# ### Set optimizer

# +
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# -

# ### Train network

for epoch in range(10):
    running_loss = 0.0
    for i, (data, target) in enumerate(load_trainset()):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
            inputs = data
            labels = target

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}] Loss:\t\t{round(running_loss / 100, 3)}')
                running_loss = 0.0
print('Finished Training!')

# ### Test network

# +
correct = 0
total = 0
with torch.no_grad():
    for data, target in load_testset():
        images, labels = data, target
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
# -

# ### Test network per class

# +
nb_classes = 3

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(load_testset()):
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
labels = ['english', 'skorean', 'mandarin']
results = (confusion_matrix.diag()/confusion_matrix.sum(1)).tolist()

for label, result in zip(labels, results):
    print(f"Accuracy of the network on {label}:\t{round(result*100, 1)}%")
