import os
from random import shuffle
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

_exp_name = "Seedings_Classification_Residual_Network"
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


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)

test_tfm = transforms.Compose([
    transforms.RandomResizedCrop((128, 128)),
    transforms.Resize((128, 128)),
    transforms.GaussianBlur(7,3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

model = Residual_Network().to(device)
model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model.eval()
pred_label=[]

for imgs in os.listdir("../warmup/test"):
    domain = os.path.abspath("../warmup/test")
    path = os.path.join(domain,imgs) 
    img = Image.open(path).convert('RGB')    
    img = test_tfm(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad(): 
        output=model(img)
    pred = output.data.max(dim = 1, keepdim = True)[1]
    pred_label.append(int(pred))

print(pred_label)
Species = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']

df = pd.DataFrame()
df["file"] = os.listdir("../warmup/test")

label_list=[]
for i in pred_label :
    label_list.append(Species[i])
print(label_list)
df["species"] = label_list

df.to_csv("../warmup/submission_Residual_Network.csv",index = False)

