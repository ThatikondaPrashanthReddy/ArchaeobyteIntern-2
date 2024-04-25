import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastai.basics import*
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from glob import glob


if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
class BrainMRIDataset(Dataset):

  def __init__(self,data_dir,reshape=True,height=128,width=128,autoencoder =False):

    self.dataDirectory = data_dir
    self.no_class = glob(data_dir+'/no/*')
    self.yes_class = glob(data_dir+'/yes/*')

    self.height = height
    self.width = width
    self.reshape = reshape
    self.autoencoder = autoencoder

    labels = [0 for i in range(len(self.no_class))]
    labels += [1 for i in range(len(self.yes_class))]

    image_links = self.no_class + self.yes_class

    self.dataframe = pd.DataFrame({"image":image_links, "labels":labels})
    self.dataframe.reset_index(inplace = True ,drop=True)

  def __len__(self):
    return len(self.no_class)+len(self.yes_class)

  def __getitem__(self,idx):

    image_list = self.dataframe["image"][idx]
    label_list = self.dataframe["labels"][idx]

    if type(image_list) == str:
      image_list = [image_list]

    if not isinstance(label_list,np.int64):
      label_list = label_list.values

    image_array = []

    for image in image_list:
      image = Image.open(image).convert("L")

      if self.reshape:
        image = image.resize((self.height,self.width))

      array = np.asarray(image)

      array = array.reshape(1,self.height,self.width)

      image_array.append(array)



    return [torch.tensor(image_array,device=device),torch.tensor(label_list,device=device)]



  def __repr__(self):
    return str(self.dataframe.head(10))
dataset = BrainMRIDataset("/content/drive/MyDrive/Brain_Tumor/Brain_Tumor/brain_tumor_dataset")

class BrainTumorModel(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(1,256,kernel_size=3), #126*126*256
        nn.MaxPool2d(2,2), # 63*63*256

        nn.Conv2d(256,32,kernel_size=2) #63-2+1 = 62*62*32
    )

    # n-f+2p/s +1

    self.linear1 = nn.Linear(62,128)
    self.linear2 = nn.Linear(128,64)
    self.flat = nn.Flatten(1)
    self.linear3 = nn.Linear(126976,2)


  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    x = self.flat(x)
    x = self.linear3(x)

    return x
  
  model = BrainTumorModel()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs =4
batch_size = 32
loss_list = []

for epoch in range(epochs):
  total_loss = 0.0

  for n in range(len(dataset)//batch_size):

    data , target = dataset[n*batch_size : (n+1)*batch_size]

    ypred = model.forward(data.float())
    loss = loss_fn(ypred,target)

    total_loss+=loss

    optimizer.zero_grad() #clear the gradients
    loss.backward() # calculate the gradeint
    optimizer.step() # Wn = Wo - lr* gradeint

  loss_list.append(total_loss/batch_size)

  print("Epochs {}  Training Loss {:.2f}".format(epoch+1,total_loss/n))

  with torch.no_grad():
  fig=plt.figure(figsize=(10,10))
  plt.xlabel("Epochs")
  plt.ylabel("Loss")

#plt.plot(list(range(epochs)),loss_list)
  plt.title("loss vs epochs")
  plt.plot(loss_list)
  plt.show()

  mapping = {0:"NO",1:"Yes"}
fig = plt.figure(figsize=(20,20))

for i in range(5):
  data,target = dataset[i]
  pred = model.forward(data.float())

  pred = torch.argmax(pred,dim=1)
  plt.subplot(5,5,i+1)
  plt.imshow(data[0][0].cpu())
  plt.title(f"Actual : {mapping[target.cpu().detach().item()]} Prediction : {mapping[pred.cpu().detach().item()]}")
  plt.show()