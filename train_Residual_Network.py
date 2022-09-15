import os
from random import shuffle
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

_exp_name = "Seedings_Classification_Residual_Network"

myseed = 7777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#Transforms
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop((128, 128)),
    transforms.Resize((128, 128)),
    transforms.GaussianBlur(7,3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])

train_dataset = ImageFolder(root='../warmup/train',transform=train_tfm)
#print(train_dataset.imgs)
print("="*50)
print(train_dataset.class_to_idx.items())

#split train/validation set
train_set_size = int(len(train_dataset)*0.8)
valid_set_size = len(train_dataset) - train_set_size
train_set,valid_set = torch.utils.data.random_split(train_dataset,[train_set_size,valid_set_size])
print("="*50)
print(len(train_set))
print(train_set[500])
print("="*50)
print(len(valid_set))
print(valid_set[120])
print("="*50)

#DataLoader
batch_size = 32
train_loader = DataLoader(train_set,batch_size = batch_size,shuffle=True,num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()


        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), #(64*128*128)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2,0), #(64*64*64)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), #(128*64*64)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0), #(128*32*32)
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), #(256*32*32)
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 0), #(256*16*16)
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), #(512*16*16)
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, 0), #(512*8*8)
        )

        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), #(512*8*8)
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, 0), #(512*4*4)

        )

        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), #(512*4*4)
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, 0), #(512*2*2)

        )

        self.size_adj_1 =  nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.MaxPool2d(2,2,0),
        )

        self.size_adj_3 =  nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.MaxPool2d(2,2,0),
        )

        self.MaxPool = nn.MaxPool2d(2, 2, 0)

        self.ReLU = nn.ReLU()

        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 12)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 12]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)

        tmp = self.size_adj_1(x1)

        x2 = self.cnn_layer2(x1)

        x2 = x2 + tmp 

        x2 = self.ReLU(x2)


        x3 = self.cnn_layer3(x2)

        tmp = self.size_adj_3(x3)

        x4 = self.cnn_layer4(x3)

        x4 = x4 + tmp

        x4 = self.ReLU(x4)


        x5 = self.cnn_layer5(x4)

        tmp = self.MaxPool(x5)

        x6 = self.cnn_layer6(x5)

        x6 = x6 + tmp

        x6 = self.ReLU(x6)

        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x6.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout


# "cuda" only when GPUs are available.
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)


# The number of training epochs and patience.
n_epochs = 600
patience = 300  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
model = Residual_Network().to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

#plot function(Data type:list)
def plot_img(epoch, valid_loss, valid_accs, train_loss, train_accs):
    epochs = [i for i in range(epoch + 1)]
    plt.plot(epochs, train_loss, label="Train_Loss")
    plt.plot(epochs, valid_loss, label="Valid_Loss")
    plt.legend()
    plt.title("Loss_fig")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig("./Loss_Residual_Network.png")
    plt.cla()

    plt.plot(epochs, train_accs, label="Train_acc")
    plt.plot(epochs, valid_accs, label="Valid_acc")
    plt.legend()
    plt.title("acc_fig")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig('./Accs_Residual_Network.png')
    plt.cla()

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
train_loss_list = []
train_accs_list = []
valid_loss_list = []
valid_accs_list = []
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc.cpu())

    train_loss_list.append(sum(train_loss) / len(train_loss))
    train_accs_list.append(sum(train_accs) / len(train_accs))

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss_list[-1]:.5f}, acc = {train_accs_list[-1]:.5f}")
    with open(f"./{_exp_name}_log.txt", "a") as log_file:
            log_file.write(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss_list[-1]:.5f}, acc = {train_accs_list[-1]:.5f}\n")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc.cpu())
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss_list.append(sum(valid_loss) / len(valid_loss))
    valid_accs_list.append(sum(valid_accs) / len(valid_accs))

    plot_img(epoch, valid_loss_list, valid_accs_list, train_loss_list, train_accs_list)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_list[-1]:.5f}, acc = {valid_accs_list[-1]:.5f}")

    # update logs
    if valid_accs_list[-1] > best_acc:
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_list[-1]:.5f}, acc = {valid_accs_list[-1]:.5f} -> best")
        with open(f"./{_exp_name}_log.txt", "a") as log_file:
            log_file.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_list[-1]:.5f}, acc = {valid_accs_list[-1]:.5f} -> best\n")
    else:
        with open(f"./{_exp_name}_log.txt", "a") as log_file:
            log_file.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_list[-1]:.5f}, acc = {valid_accs_list[-1]:.5f}\n")

    # save models
    if valid_accs_list[-1] > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = valid_accs_list[-1]
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break