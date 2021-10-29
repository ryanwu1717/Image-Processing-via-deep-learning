#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, glob, time, copy, random, zipfile
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
    
classes = list()
with open('./2021VRDL_HW1_datasets/classes.txt') as f:
   for line in f:
       # For Python3, use print(line)
        classes.append(line[:-1])
        if 'str' in line:
            break
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()


# In[2]:


torch.__version__
train_dir = './2021VRDL_HW1_datasets/training_images'
test_dir = './2021VRDL_HW1_datasets/training_images'

os.listdir(train_dir)[:5]
os.listdir(test_dir)[:5]


# In[3]:


train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
img = Image.open(train_list[2])
plt.imshow(img)
plt.axis('off')
plt.show()
img = Image.open(test_list[0])
plt.imshow(img)
plt.axis('off')
plt.show()


# In[4]:


class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
               transforms.Resize(255),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# In[5]:


# trasform
class newImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                   torchvision.transforms.RandomOrder([
                   torchvision.transforms.RandomCrop((256, 256)),
                   torchvision.transforms.RandomHorizontalFlip(),
                   torchvision.transforms.RandomVerticalFlip()
                ]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                torchvision.transforms.CenterCrop((256, 256)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# In[6]:


# build label_dict
label_dict = {}
with open('./2021VRDL_HW1_datasets/training_labels.txt', 'r', encoding='utf-8-sig') as R:
    lines = R.readlines()
for line in lines:
    line = line.split('\n')[0].split(" ")
    label_dict[line[0]] = (line[1])
label_dict


# In[7]:


# spilt train data and test data 
train_set = train_list
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])
print('Train data set:', len(train_set))
test_list = valid_set
print('Valid data set:', len(valid_set))


# In[8]:


# Dataset
class PartDataset(data.Dataset):
    
    def __init__(self, label_dict, file_list, transform=None, phase='train'):
        self.label_dict = label_dict
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        
        img_transformed = self.transform(img, self.phase)
        # Get Label
        label = self.label_dict[img_path.split('\\')[-1]]
        return img_transformed, label


# In[9]:


# setting
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# mean = (0.5, 0.5, 0.5)
# std = (0.5, 0.5, 0.5)

batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[10]:


train_dataset = PartDataset(label_dict, train_list, transform=newImageTransform(size, mean, std), phase='train')
test_dataset = PartDataset(label_dict, test_list, transform=newImageTransform(size, mean, std), phase='test')


# In[11]:


# DataLoader
train_dataloader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=16, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'test': test_dataloader}

# Operation Check
print('Operation Check')
batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label)


# In[12]:


# build net with pretrain modal
pretrained = True
num_classes = 200


net =models.resnet50( pretrained=pretrained).to(device)
num_ftrs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes)
)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 200)

net


# In[13]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)


# In[14]:


import time
def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    net = net.to(device)
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)
        
        for phase in ['train', 'test']:
            
            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
#                 print(type(labels))
               
                labelsArr = []
                
                for tmplabel in (labels):
#                     print(tmplabel)
                    labelsArr.append(int(tmplabel.split(".")[0])-1)
                labels = torch.tensor(labelsArr).to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
          
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                print(time.time())
                torch.save(net.state_dict(), 'best_checkpoint_last.pth')
                
                xtrain.append(epoch + 1)
                ytrainAcc.append(epoch_acc)
                ytrain.append(epoch_loss)
            else:
                xtest.append(epoch + 1)
                ytestAcc.append(epoch_acc)
                ytest.append(epoch_loss)
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


# In[15]:


import matplotlib.pyplot as plt
xtrain=[]
ytrainAcc=[]
ytrain=[]

xtest=[]
ytestAcc=[]
ytest=[]


# In[16]:


# train
num_epoch = 60
net = train_model(net, dataloader_dict, criterion, optimizer, num_epoch)


# In[17]:


# check  convergence
plt.plot(xtrain, ytrain, label = "train")
plt.plot(xtest, ytest, label = "test")
plt.xlabel('loss')
plt.xlabel('epoch')
plt.legend()
 
plt.show()


# In[18]:


# save the checkpoint
checkpoint = net.load_state_dict(torch.load('./best_checkpoint_last.pth'))
net.to(device)
net.eval()


# In[19]:


# load the checkpoint
testorder = list()
with open('./2021VRDL_HW1_datasets/testing_img_order.txt') as f:
   for line in f:
       # For Python3, use print(line)
        testorder.append(line.split('\n')[0])
        if 'str' in line:
            break
print(testorder)


# In[20]:


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


# In[21]:


# predict and make file
norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)
# norm_mean = [0.5, 0.5, 0.5]
# norm_std = [0.5, 0.5, 0.5]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

import time
path = 'answer.txt'
f = open(path, 'w')

with torch.no_grad():
#         print(classes)
        for  img_name in testorder:
            print(img_name)
            path_img ='./2021VRDL_HW1_datasets/testing_images/'+img_name
            img_rgb = Image.open(path_img).convert('RGB')

            img_tensor = img_transform(img_rgb, inference_transform)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.to(device)
            
            # step 3/4 : tensor --> vector
            time_tic = time.time()
            outputs = net(img_tensor)
            time_toc = time.time()
            
             # step 4/4 : visualization
            _, pred_int = torch.max(outputs.data, 1)
#             print(int(pred_int))
            if(pred_int >= 200):
                pred_int=0
                print(int(pred_int))
            
#             print(classes[int(pred_int)])
            pred_str = classes[int(pred_int)]
        
            f.write(img_name)           
            f.write(' ')
            f.write(pred_str)  
            f.write('\n')
            
f.close()


# In[ ]:




