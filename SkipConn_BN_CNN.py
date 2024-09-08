#!/usr/bin/env python
# coding: utf-8



#import libraries that are used in the code below
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as tvt
from PIL import Image
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import random
import time
import numpy
import cv2
from pycocotools.coco import COCO





#Custom dataset class for image classification using COCO subset dataset same as homework 4
class MyCocoDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.file_names = []  #storing filename paths for images and associated labels
        self.classes = ['boat', 'cake', 'couch', 'dog', 'motorcycle']
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for label in self.classes:
            folder_path = os.path.join(root, label)
            for image_file in os.listdir(folder_path):
                info = {'image_path': os.path.join(folder_path, image_file), 'label': self.classes.index(label)} #creating a dictionary to store image path and label for every image
                self.file_names.append(info) #creating a list of dictionary

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = file_name['image_path']

        # Reading image as a PIL image
        image = Image.open(image_path)

        # Applying predefined transform
        augmented_image = self.transform(image)

        # Setting label 
        label = file_name['label']

        return augmented_image, label




#creating instances for train and validation dataset classes, using the same dataset from homework 4
trainDataset = MyCocoDataset('/Users/skose/Downloads/CocoSubset/train') 
valDataset = MyCocoDataset('/Users/skose/Downloads/CocoSubset/val')



#creating instances of Train and validation dataloader
trainDataLoader = DataLoader(trainDataset, batch_size = 4,shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size = 4,shuffle=True)





# The code is derived from Dr. Avinash Kak's DL Studio Module for the training loop and is same as my homework 4 training routine
def training_routine(net, epochs, trainDataLoader, device, model_path):
    # Initialize a list to store training losses during each epoch
    training_loss = []
    # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.99))

    # Move the network to the specified device (CPU or GPU)
    net = net.to(device)

    # Set the network in training mode
    net.train()

    print("Begin Training...\n")

    # Training loop over epochs
    for epoch in range(epochs):
        running_loss = 0.0

        # Iterate over batches in the training data loader
        for i, data in enumerate(trainDataLoader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients to accumulate new gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs through the network
            outputs = net(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update the parameters (weights) of the model
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Print running loss every 100 batches
            if (i + 1) % 100 == 0:
                print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                training_loss.append(running_loss / 100)
                running_loss = 0.0

    print("Finished Training!\n")

    # Save the trained model state dictionary to the specified path
    torch.save(net.state_dict(), model_path)

    # Return the list of training losses to plot the loss curve
    return training_loss


 


#This testing loop derives ideas from the previous year's homework 4 assignment at https://engineering.purdue.edu/DeepLearn/2_best_solutions/2023/Homeworks/HW4/2BestSolutions/1.pdf
def model_evaluation(net, valDataLoader, batch_size, device, desired_classes, model_path):
    # The part of code is derived from Dr. Avinash Kak's DL Studio Module for testing with small changes in it.
    print("Begin...\n")

    # Loading trained model state
    net.load_state_dict(torch.load(model_path))

    # Set the network in evaluation mode
    net.eval()

    # Initialize variables for accuracy calculation
    correct, total = 0, 0

    # Initialize a confusion matrix
    confusion_matrix = torch.zeros(5, 5)

    # Initialize variables for per-class accuracy calculation
    class_correct = [0] * 5
    class_total = [0] * 5

    with torch.no_grad():
        # Iterate over batches in the testing data loader
        for i, data in enumerate(valDataLoader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: compute predicted outputs by passing inputs through the network
            output = net(inputs)

            # Find the predicted class (index with maximum probability)
            _, predicted = torch.max(output.data, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update confusion matrix
            comp = predicted == labels
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1

            # Update per-class accuracy
            for j in range(batch_size):
                label = labels[j]
                class_correct[label] += comp[j].item()
                class_total[label] += 1
    confusion_matrix_np = confusion_matrix.cpu().numpy()
    # Print per-class accuracy
    for j in range(5):
        print('Prediction accuracy for %5s : %2d %%' % (desired_classes[j], 100 * class_correct[j] / class_total[j]))

    print("Finished!\n")

    # Print overall accuracy
    print('Accuracy of the network on validation images: {}%'.format(100 * float(correct / total)))

    # Return confusion matrix and overall accuracy
    return confusion_matrix_np, float(correct / total)


 


#set the device value to the currently used device
if torch.backends.mps.is_available() == True:
 device = torch.device("mps")
else:
 device = torch.device("cpu")
# Print out currently using device
print("Currently using: {}".format(device))





#function to plot the training loss curve for the network (the depth and learning rate values were changed each time while plotting the graphs for every network)
def train_loss_plot(train_loss, title):
    plt.plot(train_loss[0], 'b', label='Depth 32, Learning rate: 1e-5')
    plt.title(title)
    plt.legend(loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


 


#function to plot the confusion matrix for the network (the depth and learning rate values were changed each time while plotting the graphs for every network)
def plot_confusion_matrix(confusion_matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    class_names = ['boat', 'cake', 'couch', 'dog', 'motorcycle']
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Display values in the matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

    plt.ylabel('Ground Truth Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

 

#This code is borrowed directly from the DL studio module code provided by Professor Kak for skip connection class using skip block and BMEnet class
class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if downsample:
            self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

    def forward(self, x):
        identity = x
        out = self.convo1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.convo2(out)
        out = self.bn2(out)
        out = nn.functional.relu(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                out = out + identity
            else:
                out = out + torch.cat((identity, identity), dim=1)
        return out

class HW5Net(nn.Module):
    def __init__(self, skip_connections=True, depth=32):
        super().__init__()
        if depth not in [8, 16, 32, 64]:
            sys.exit("BMEnet has been tested for depth for only 8, 16, 32, and 64")
        self.depth = depth // 8
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.skip64_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip64_arr.append(SkipBlock(64, 64, skip_connections=skip_connections))
        self.skip64ds = SkipBlock(64, 64, downsample=True, skip_connections=skip_connections)
        self.skip64to128 = SkipBlock(64, 128, skip_connections=skip_connections)
        self.skip128_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip128_arr.append(SkipBlock(128, 128, skip_connections=skip_connections))
        self.skip128ds = SkipBlock(128, 128, downsample=True, skip_connections=skip_connections)
        
        # The following declaration is predicated on the assumption that the number of
        # output nodes (CxHxW) from the final convo layer is exactly 2048 for each
        # input image. Depending on the size of the input image, this places a constraint
        # on how many downsampling instances of SkipBlock you can call in a network.
        self.fc1 = nn.Linear(8192, 1000) # input size = 8192 for 
        self.fc2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))          
        for i, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            x = skip64(x)                
        x = self.skip64ds(x)
        for i, skip64 in enumerate(self.skip64_arr[self.depth//4:]):
            x = skip64(x)                
        x = self.skip64ds(x)
        x = self.skip64to128(x)
        for i, skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            x = skip128(x)                
        for i, skip128 in enumerate(self.skip128_arr[self.depth//4:]):
            x = skip128(x)                
        # See the comment block above the "self.fc1" declaration in the constructor code.
        x = x.view(x.shape[0], - 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


 


#run the training loop for depth =32 and learning rate = 1e-3 and save the training loss for plotting later
net= HW5Net(skip_connections=True, depth=32)
epochs_net = 15
model_path=os.path.join('/Users/skose/Downloads/path', 'nethw5.pth')
training_loss_net = training_routine(net, epochs_net, trainDataLoader, device, model_path)


 


desired_classes=['boat', 'cake', 'couch', 'dog', 'motorcycle']


 


#run the evaluation on validation dataset for depth =32 and learning rate = 1e-3 and save the validation accuracy and confusion matrix 
batch_size=4
confusion_matrix_net, validation_acc_net = model_evaluation(net, valDataLoader, batch_size, device, desired_classes,model_path)


 


#Plot confusion matrix  for depth =32 and learning rate = 1e-3
title= 'Confusion matrix for Hw5Net depth = 32 and learning rate = 1e-3'
plot_confusion_matrix(confusion_matrix_net, title)


 


# training loss for depth =32 and learning rate = 1e-3
train_loss = []
train_loss.append(training_loss_net)
train_loss_plot(train_loss, 'Training Loss for Network')


 


#Total number of layers for the network 
num_layers = len( list ( net . parameters () ) )
num_layers


 


#run the training loop  for depth =32 and learning rate = 1e-4 and save the training loss for plotting later
net= HW5Net(skip_connections=True, depth=32)
epochs_net = 15
model_path=os.path.join('/Users/skose/Downloads/path', 'nethw55.pth')
training_loss_net2 = training_routine(net, epochs_net, trainDataLoader, device, model_path)


 


#run the evaluation on validation dataset for depth =32 and learning rate = 1e-4 and save the validation accuracy and confusion matrix 
batch_size=4
confusion_matrix_net2, validation_acc_net2 = model_evaluation(net, valDataLoader, batch_size, device, desired_classes,model_path)


 

#Plot confusion matrix  for depth =32 and learning rate = 1e-4
title= 'Confusion matrix for Hw5Net depth = 32 and learning rate = 1e-4'
plot_confusion_matrix(confusion_matrix_net2, title)


 


#train loss for depth = 32 and learning rate = 1e-4
train_loss = []
train_loss.append(training_loss_net2)
train_loss_plot(train_loss, 'Training Loss for Network')


 


#run the training loop  for depth =64 and learning rate = 1e-4 and save the training loss for plotting later
net= HW5Net(skip_connections=True, depth=64)
model_path=os.path.join('/Users/skose/Downloads/path', 'nethw53.pth')
training_loss_net3 = training_routine(net, epochs_net, trainDataLoader, device, model_path)


 


#run the evaluation on validation dataset  for depth =64 and learning rate = 1e-4 and save the validation accuracy and confusion matrix 
batch_size=4
confusion_matrix_net3, validation_acc_net3 = model_evaluation(net, valDataLoader, batch_size, device, desired_classes,model_path)


 


#Plot confusion matrix  for depth =64 and learning rate = 1e-4
title= 'Confusion matrix for Hw5Net depth = 64 and learning rate = 1e-4'
plot_confusion_matrix(confusion_matrix_net2, title)


 


#train loss for depth 64 and learning rate 1e-4
train_loss = []
train_loss.append(training_loss_net3)
train_loss_plot(train_loss, 'Training Loss for Network')


 


#This code is borrowed directly from the last code provided by Professor Kak over email with the new modifications for skip block
#Tried to run this code for different learning rates in order to just compare whether the performances improves as compared 
#to my previous implementation

class SkipBlock(nn.Module):
          
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1 = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.in2out = nn.Conv2d(in_ch, out_ch, 1)
        
        if downsample:
            self.downsampler1 = nn.Conv2d(in_ch, in_ch, 1, stride=2)
            self.downsampler2 = nn.Conv2d(out_ch, out_ch, 1, stride=2)

    def forward(self, x):
        identity = x                                     
        out = self.convo1(x)                              
        out = self.bn1(out)                              
        out = F.relu(out)
        out = self.convo2(out)                              
        out = self.bn2(out)                              
        out = F.relu(out)
        
        if self.downsample:
            identity = self.downsampler1(identity)
            out = self.downsampler2(out)
            
        if self.skip_connections:
            if (self.in_ch == self.out_ch) and (not self.downsample):
                out = out + identity
            elif (self.in_ch != self.out_ch) and (not self.downsample):
                identity = self.in2out(identity)
                out = out + identity
            elif (self.in_ch != self.out_ch) and self.downsample:
                out = out + torch.cat((identity, identity), dim=1)
                
        return out

class BMEnet(nn.Module):
    """
    Class Path: DLStudio -> SkipConnections -> BMENet
    """
    def __init__(self, skip_connections=True, depth=8):
        super(BMEnet, self).__init__()
        self.depth = depth
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.skip64_arr = nn.ModuleList([SkipBlock(64, 64, skip_connections=skip_connections) for _ in range(depth)])
        self.skip64to128ds = SkipBlock(64, 128, downsample=True, skip_connections=skip_connections)
        self.skip128_arr = nn.ModuleList([SkipBlock(128, 128, skip_connections=skip_connections) for _ in range(depth)])
        self.skip128to256ds = SkipBlock(128, 256, downsample=True, skip_connections=skip_connections)
        self.skip256_arr = nn.ModuleList([SkipBlock(256, 256, skip_connections=skip_connections) for _ in range(depth)])
        self.fc1 = nn.Linear(16384, 1000)
        self.fc2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2, 2)

        
        for skip64 in self.skip64_arr:
            x = skip64(x)                
        x = self.skip64to128ds(x)
        
        for skip128 in self.skip128_arr:
            x = skip128(x)                
        x = self.skip128to256ds(x)
        
        for skip256 in self.skip256_arr:
            x = skip256(x)                
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


 


#run the training loop for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-5 
#and save the training loss for plotting later
netNew= BMEnet(skip_connections=True, depth=32)
epochs_net = 15
model_path6=os.path.join('/Users/skose/Downloads/path', 'nethw6.pth')
training_loss_net6 = training_routine(netNew, epochs_net, trainDataLoader, device, model_path6)


 


#run the evaluation on validation dataset for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-5 
#and save the validation accuracy and confusion matrix 
batch_size=4
confusion_matrix_net6, validation_acc_net6 = model_evaluation(netNew, valDataLoader, batch_size, device, desired_classes,model_path6)
#Plot confusion matrix 
title= 'Confusion matrix for BMEnet with Dr. Kak new code'
plot_confusion_matrix(confusion_matrix_net6, title)
#training loss for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-5
train_loss = []
train_loss.append(training_loss_net6)
train_loss_plot(train_loss, 'Training Loss for Network')


 


#run the training loop for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-4 
#and save the training loss for plotting later
netNew1= BMEnet(skip_connections=True, depth=32)
epochs_net = 15
model_path7=os.path.join('/home/skose', 'nethw7.pth')
training_loss_net7 = training_routine(netNew1, epochs_net, trainDataLoader, device, model_path7)


 


#run the evaluation on validation dataset for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-4 and save the validation accuracy and confusion matrix 
batch_size=4
confusion_matrix_net7, validation_acc_net7 = model_evaluation(netNew1, valDataLoader, batch_size, device, desired_classes,model_path7)
#Plot confusion matrix 
title= 'Confusion matrix for BMEnet with Dr. Kak new code'
plot_confusion_matrix(confusion_matrix_net7, title)
#training loss for the borrowed BMEnet code from Dr. Kak's email for depth 32 and learning rate 1e-4
train_loss = []
train_loss.append(training_loss_net7)
train_loss_plot(train_loss, 'Training Loss for Network')

