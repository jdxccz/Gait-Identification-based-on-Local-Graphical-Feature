import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
import os
import pickle
import sys
import random
import time

root = "C:\\Users\\zxk\\Desktop\\capstone\\prime-joints\\"
checkpoint_path = "C:\\Users\\zxk\\Desktop\\capstone\\prime-joints\\checkpoint\\test\\"
view = "090"
train_bound = 74
checkpoint_name = "epoch_700_m_2022-04-01-16-59-16_0.1321.pth.tar"
p_kinds = ["nm","bg","cl"]

class sgDataset(Dataset):
    def __init__(self,root,train_bound,view):
        gallery_histogram_p = root+"histogram\\"+str(train_bound+1) + "_124_gallery_"+view+"_histogram.txt"
        gallery_index_p = root+"index\\"+str(train_bound+1) +"_124_gallery_"+view+".txt"
        probe_histogram_p = root+"histogram\\"+str(train_bound+1) + "_124_probe_"+view+"_histogram.txt"
        probe_index_p = root+"index\\"+str(train_bound+1) +"_124_probe_"+view+".txt"
        g_labels,p_labels,labels = [],[],[]
        # g_d = torch.tensor([],dtype=float)
        self.x_data = torch.tensor([],dtype=float)
        y_f = []


        with open(gallery_histogram_p, 'rb') as ffile:
            gallery_histograms = pickle.load(ffile)
            g_d = torch.from_numpy(gallery_histograms)
        
        with open(gallery_index_p,'r') as ffile:
            lines = ffile.readlines()
            for line in lines:
                i = line.find("operator")
                label = int(line[i+9:i+12])
                g_labels.append(label)
        
        N,H,L = g_d.size()
        g_d = g_d.reshape(N,1,H,L)

        with open(probe_histogram_p, 'rb') as ffile1, open(probe_index_p,'r') as ffile2:
            lines = ffile2.readlines()
            probe_histograms = pickle.load(ffile1)
            p_d = torch.from_numpy(probe_histograms)
            p_d = torch.unsqueeze(p_d,dim=1)
            for i in range(len(lines)):
                p= line.find("operator")
                p_label = int(lines[i][p+9:p+12])
                y_list = []
                x_t = torch.tensor([],dtype=float)
                p_kind = p_kinds.index(lines[i][p+13:p+15])

                for j in range(N):
                    tmp_x = torch.cat([g_d[j,::],p_d[i,::]],dim=0)
                    tmp_x = torch.unsqueeze(tmp_x,dim=0)
                    # print(p_label,g_labels[j])
                    if p_label == g_labels[j]:
                        tmp_y = [1,p_kind]
                    else:
                        tmp_y = [0,p_kind]
                    y_list.append(tmp_y)
                    x_t = torch.cat([x_t,tmp_x],dim=0)
                
                y_f.append(y_list)
                x_t = torch.unsqueeze(x_t,dim=0)
                self.x_data = torch.cat([self.x_data,x_t],dim=0)

        self.y_data = torch.tensor(y_f)
        self.len = self.y_data.shape[0]
        if (self.x_data.shape[0] != self.y_data.shape[0]):
            sys.exit("data num and label num should be the same!")
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

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

train_data_set = sgDataset(root,train_bound,view)
train_loader = DataLoader(dataset=train_data_set,batch_size=4,shuffle=False,num_workers=0)
psn = PSN()
psn.eval()
checkpoint_names = os.listdir(checkpoint_path)
for checkpoint_name in checkpoint_names:
    checkpoint = torch.load(checkpoint_path+checkpoint_name)
    psn.load_state_dict(checkpoint['state_dict'])

    t_num,sum = torch.tensor([0,0,0]),torch.tensor([0,0,0])
    for step, (b_x, b_y) in enumerate(train_loader):
        for i in range(b_x.shape[0]):
            input = b_x[i,::].float()
            pred = psn(input)
            label = b_y[i,::]
            p_labels = pred.argmin(dim = 0)
            sum[label[p_labels[0],[1]]] = sum[label[p_labels[0],[1]]] + 1
            if label[p_labels[0],[0]] == 1:
                t_num[label[p_labels[0],[1]]] = t_num[label[p_labels[0],[1]]] + 1
    accuracy = torch.round(t_num/sum*100)
    print(checkpoint_name,"accuracy:",accuracy)

