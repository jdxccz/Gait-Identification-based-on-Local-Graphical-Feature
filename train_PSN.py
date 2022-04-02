from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
# from dataset import LiverDataset, LiverDatasetTest,LiverDatasetExtract
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
import pickle
import sys
import random
import time

learning_rate = 1e-4
_MAX = 99
launchTimestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
train_bound = 74
_epoch = 200
epoch_id = 0 #default
root = "C:\\Users\\zxk\\Desktop\\capstone\\prime-joints\\"
checkpoint_path = "C:\\Users\\zxk\\Desktop\\capstone\\prime-joints\\checkpoint\\"
checkpoint_name = "epoch_600_m_2022-04-01-13-47-15_0.1539.pth.tar"

class DealDataset(Dataset):
    def __init__(self,root,train_bound):
        views = ['000','018','036','054','072','090','108','126','144','162','180']
        train_histogram = root+"histogram\\"+"0_"+str(train_bound) +"_train_"
        train_index = root+"index\\"+"0_"+str(train_bound) +"_train_"
        labels = []
        self.x_data = torch.tensor([],dtype=float)

        for view in views:
            with open(train_histogram+view+'_histogram.txt', 'rb') as ffile:
                view_histograms = pickle.load(ffile)
                x = torch.from_numpy(view_histograms)
                self.x_data = torch.cat([self.x_data, x], dim=0)
            
            with open(train_index+view+'.txt','r') as ffile:
                lines = ffile.readlines()
                for line in lines:
                    i = line.find("operator")
                    label = int(line[i+9:i+12])
                    labels.append([label])

        self.y_data = torch.tensor(labels)
        self.len = self.y_data.shape[0]
        if (self.x_data.shape[0] != self.y_data.shape[0]):
            sys.exit("data num and label num should be the same!")
    
    def __getitem__(self, index):
        i = index
        flag_p,flag_n = 0,0
        while flag_p == 0 or flag_n == 0:
            while i == index:
                i = random.randint(0,self.len-1)
            if self.y_data[index] == self.y_data[i]:
                if flag_p == 0:
                    x_p = torch.cat([self.x_data[index],self.x_data[i]],dim = 0)
                    flag_p = 1
                else:
                    i = index
            else:
                if flag_n == 0:
                    x_n = torch.cat([self.x_data[index],self.x_data[i]],dim = 0)
                    flag_n = 1
                else:
                    i = index
        
        x = torch.cat([x_p,x_n],dim = 0)
        y = torch.tensor([1,0])
        return x,y



        #return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class PSN(nn.Module):
    def __init__(self):

        _channel = 8
        _length = 256*5
        super().__init__()

        self.concatenated_stream = nn.Sequential(
            nn.Conv2d(2, _channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_channel, _channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_channel, _channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_channel, 1, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2)
            )

        self.fc = nn.Sequential(
            nn.Linear(_length,_length),
            nn.BatchNorm1d(_length),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(_length, 2),
            nn.BatchNorm1d(2)
            )
    
    def forward(self, x):
        N, C, H, L=x.size()
        x = x.reshape(N,C,1,H*L)
        out = self.concatenated_stream(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ",device)

loss_sum = 0
loss_count = 0
losses = []
lossMIN = _MAX

psn = PSN()
psn.train()
checkpoint = torch.load(checkpoint_path+checkpoint_name)

train_data_set = DealDataset(root,train_bound)
train_loader = DataLoader(dataset=train_data_set,batch_size=256,shuffle=True,num_workers=0)
optimizer = optim.Adam(psn.parameters(),lr=learning_rate)
_loss = nn.CrossEntropyLoss()

optimizer.load_state_dict(checkpoint['optimizer'])


for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()


psn.load_state_dict(checkpoint['state_dict'])
epoch_id = checkpoint['epoch']
psn.to(device)
print(psn)

for epoch in range(epoch_id+1,epoch_id+_epoch+1):

    for step, (b_x, b_y) in enumerate(train_loader):

        # N,H,L = b_x.size()
        # N2,L2 = b_y.size()
        # b_x = b_x.reshape(N,1,H,L)
        # b_y = b_y.reshape(N2,L2,1)
        # b_x_cyc,b_y_cyc = b_x,b_y
        # f_x,f_y = torch.tensor([]),torch.tensor([])
        # for i in range(N):
        #     b_x_cyc = torch.roll(b_x_cyc,1,dims=0)
        #     b_y_cyc = torch.roll(b_y_cyc,1,dims=0)
        #     tmp_x = torch.cat([b_x,b_x_cyc],dim=1)
        #     tmp_y = torch.cat([b_y,b_y_cyc],dim=2)
        #     f_x = torch.cat([f_x,tmp_x],dim=0)  
        #     f_y = torch.cat([f_y,tmp_y],dim=0)   
        # fy_0 = f_y[:,:,0] - f_y[:,:,1]
        # fy_0[fy_0 != 0] = 1
        # fy_0 = fy_0.reshape(-1)

        N,H,L = b_x.size()
        f_x = b_x.reshape(N*2,2,int(H/4),L)
        fy_0 = b_y.reshape(-1)

        # print(f_x.shape,fy_0.shape)

        f_x = f_x.float().to(device)
        fy_0 = fy_0.long().to(device)

        output = psn(f_x)
        loss = _loss(output, fy_0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            if loss.item() < lossMIN:
                lossMIN = loss.item()
            loss_sum = loss_sum + loss.item()
            loss_count = loss_count + 1

    if epoch % 20 == 0:
        loss_mean = round(loss_sum/loss_count,4)
        losses.append(loss_mean)
        print("The",epoch,"Epoch Loss: ",loss_mean)
        loss_sum,loss_count = 0,0

    if epoch % 50 == 0:
        torch.save({'epoch': epoch, 'state_dict': psn.state_dict(), 'best_loss': lossMIN,
            'optimizer': optimizer.state_dict()},
            checkpoint_path + 'epoch_'+str(epoch) + "_m_" + launchTimestamp + '_' + str(round(lossMIN,4)) + '.pth.tar')

#print(losses)