# -*- coding: utf-8 -*-
"""
Author: Evan Munro
Date: December 15, 2018
Description: Contains classes for the following PyTorch Models: Multinomial logistic regression with choice-invariant coefficients
and no intercept, 3-Layer Neural Network, Neural Network with Embeddings, and Basic Recurrent Neural Network. Also contains relevant Dataset classes.
"""



import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np

# first dimension is N dimension in data loader

class MultinomialLogit(torch.nn.Module):

    def __init__(self,m):
        super(MultinomialLogit,self).__init__()
        self.linear = torch.nn.Linear(m,1,bias=False)
        self.lsmax = torch.nn.LogSoftmax(1)

    # this is defined on a batch
    def forward(self,x):
        y_pred=self.linear(x)
        y_pred=self.lsmax(y_pred)
        return y_pred.squeeze()

class NonLinearMLogit(torch.nn.Module):

    def __init__(self,n_x):

        super(NonLinearMLogit,self).__init__()

        self.linear1 = torch.nn.Linear(n_x,10,bias=False)
        self.linear2 = torch.nn.Linear(10,8,bias=False)
        self.linear3 = torch.nn.Linear(8,1,bias=False)
        self.lsmax = torch.nn.LogSoftmax(1)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.lsmax(self.linear3(x))
        return x.squeeze()

class NLMLEmbed(torch.nn.Module):

    def __init__(self,n_x,n_i,n_y,k):
        super(NLMLEmbed,self).__init__()
        self.embed_user = torch.nn.Embedding(n_i+1,k,padding_idx=n_i)
        self.embed_item = torch.nn.Embedding(n_y+1,k,padding_idx=n_y)
        self.embed_item.weight.data = torch.zeros(n_y+1,k)
        self.embed_user.weight.data = torch.zeros(n_i+1,k)
        self.nlml = NonLinearMLogit(n_x+2*k+1)
    def forward(self,x,idx):
        e_user = self.embed_user(idx[:,:,0].long())
        e_item = self.embed_item(idx[:,:,1].long())
        interaction = torch.einsum('bcij,bcj->bci',(e_user.view(e_user.size(0),e_user.size(1),1,e_user.size(2)),e_item))
        input_data = torch.cat((x,e_user,e_item,interaction),dim=2)
        y_pred = self.nlml(input_data)
        return y_pred



class ChoiceRNN(torch.nn.Module):
    def __init__(self,n_x,n_i,n_y,k):
        super(ChoiceRNN,self).__init__()
        self.embed_user = torch.nn.Embedding(n_i+1,k,padding_idx=n_i)
        self.embed_item = torch.nn.Embedding(n_y+1,k,padding_idx=n_y)
        self.linear1 = torch.nn.Linear(n_x+k*2+1+1,10,bias=False)
        self.linear2 = torch.nn.Linear(10,8,bias=False)
        self.linear3 = torch.nn.Linear(8,1,bias=False)
        self.lsmax = torch.nn.LogSoftmax(0)

    def forward(self,x,idx):
        e_user = self.embed_user(idx[:,:,0].long())
        e_item = self.embed_item(idx[:,:,1].long())
        interaction = torch.einsum('bcij,bcj->bci',(e_user.view(e_user.size(0),e_user.size(1),1,e_user.size(2)),e_item))
        input_data = torch.cat((x,e_user,e_item,interaction),dim=2)
        input_data = input_data.view(input_data.size(1),input_data.size(2))
        a0  = torch.zeros(1)
        activations = torch.ones(input_data.size(0))
        for i in range(input_data.size(0)):
            x = torch.cat((input_data[i,:],a0))
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            a0 = self.lsmax(self.linear3(x))
            activations[i] = a0
        result = self.lsmax(activations)
        return result.view(1,result.size(0))

# Not used in final project
class ChoiceCNN(torch.nn.Module):

    def __init__(self,n_x,n_i,n_y,k):
        super(ChoiceCNN,self).__init__()
        self.embed_user = torch.nn.Embedding(n_i+1,k,padding_idx=n_i)
        self.embed_item = torch.nn.Embedding(n_y+1,k,padding_idx=n_y)
        #input dimensions
        n_h = n_y
        n_w = n_x + 2*k + 1
        n_c = 1
        self.conv1 = torch.nn.Conv2d(1,1,kernel_size=(100,12),stride=2,padding=0)
        n_w = int((n_w  - 12)/2)+1
        n_h = int((n_h - 100)/2)+1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        n_w = int((n_w - 2)/2) + 1
        n_h = int((n_h - 2)/2) + 1
        self.conv2 = torch.nn.Conv2d(1,2,kernel_size=(100,12),stride=2,padding=0)
        n_c = 2
        n_w = int((n_w - 12)/2)+1
        n_h = int((n_h -100)/2)+1
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        n_w = int((n_w - 2)/2) + 1
        n_h = int((n_h - 2)/2) + 1
        print(n_w,n_h,n_c)
        self.conv_out_len = n_h*n_w*n_c
        self.fc1 = torch.nn.Linear(n_h*n_w*n_c,1000)
        self.fc2 = torch.nn.Linear(1000,n_y)
        self.lsmax = torch.nn.LogSoftmax(0)
    def forward(self,x,idx):
        e_user = self.embed_user(idx[:,:,0].long())
        e_item = self.embed_item(idx[:,:,1].long())
        interaction = torch.einsum('bcij,bcj->bci',(e_user.view(e_user.size(0),e_user.size(1),1,e_user.size(2)),e_item))
        input_data = torch.cat((x,e_user,e_item,interaction),dim=2)
        input_data  = input_data.view(input_data.size(0),1,input_data.size(1),input_data.size(2))
        x = self.conv1(input_data)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0),x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

###############################################################################
#Dataset Classes
###############################################################################

#classes for pytorch data loading
class ChoiceDataset(Dataset):
    def __init__(self, filename,id_limits,chunksize,choice_var,feature_vars):
        choiceData = pd.HDFStore(filename)
        print(choiceData.keys())
        self.chunksize = chunksize
        self.hdfStore = choiceData
        self.sessions = choiceData["sessions"]
        self.length = min(len(choiceData["sessions"]),id_limits[1]-id_limits[0]+1)
        self.n_y = len(choiceData["items"])
        self.n_x = len(feature_vars)
        self.n_i = len(choiceData["users"])
        self.choiceVar = choice_var
        self.features = feature_vars
        self.limits = id_limits
        query = "index>="+str(id_limits[0])+"&index<="+str(id_limits[0]+chunksize)
        self.data = choiceData.select("data",query)

    def __getitem__(self, idx):
        session_id = idx + self.limits[0]
        max_session_id = min(idx+self.limits[0]+self.chunksize,self.limits[0]+self.length-1)
        #load another batch of sessions into memory
        if session_id not in self.data.index:
            query = "index>=" + str(session_id)+ "&index<="+str(max_session_id)
            self.data = self.hdfStore.select("data",query)
	#select data corresponding to requested session
        session = self.data[self.data.index==session_id]
        session_x = session[self.features]
        #session_x["N_ratings_overall"].apply(np.log)
        #Check why this is the case (.item() doenst work herein dev set?)?
        label = session[session[self.choiceVar]==1].item_idx.item()

        #convert to tensor and include missing items as zeros
        data_tensor = torch.zeros(self.n_y,self.n_x,dtype=torch.float64)
        #data_tensor[:,0] = 10e7
        data_tensor[session.item_idx.values,:] = torch.tensor(session_x.values)
        missing = torch.ones(self.n_y)
        missing[session.item_idx.values] = 0
        idx = torch.ones(self.n_y,2,dtype=torch.int32)
        #give unique index to each restaurant that is missing from user's choice set
        idx[:,0] = self.n_i
        idx[:,1] = self.n_y
        idx[session.item_idx.values,:] = torch.tensor(session[["user_idx","item_idx"]].values,dtype=torch.int32)
        return (data_tensor, torch.tensor(label),idx)

    def __len__(self):
        return self.length

class ChoiceSequence(Dataset):
    def __init__(self,filename,id_limits,chunksize,choice_var,feature_vars):
        choiceData = pd.HDFStore(filename)
        self.chunksize = chunksize
        self.hdfStore  = choiceData
        self.sessions = choiceData["sessions"]
        self.length = min(len(choiceData["sessions"]),id_limits[1] - id_limits[0]+1)
        self.n_y = len(choiceData["items"])
        self.n_x = len(feature_vars)
        self.n_i = len(choiceData["users"])
        self.choiceVar = choice_var
        self.features = feature_vars
        self.limits = id_limits
        query = "index>=" + str(id_limits[0]) + "&index<="+str(id_limits[0]+chunksize)
        self.data = choiceData.select("data",query)

    def __getitem__(self,idx):
        session_id = idx+self.limits[0]
        max_session_id = min(idx+self.limits[0]+self.chunksize,self.limits[0]+self.length-1)
        if session_id not in self.data.index:
            query = "index>=" + str(session_id) + "&index<=" + str(max_session_id)
            self.data = self.hdfStore.select("data",query)
        session = self.data[self.data.index==session_id]
        session = session.sort_values(['log_distance'])
        session_x = session[self.features]
        label = np.nonzero(session[self.choiceVar]==1)[0]
        data_tensor = torch.tensor(session_x.values)
        idx = torch.tensor(session[["user_idx","item_idx"]].values,dtype=torch.int32)
        return (data_tensor, torch.tensor(label),idx)

    def __len__(self):
        return self.length
