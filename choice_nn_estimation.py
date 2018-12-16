#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:00:35 2018

@author: evanmunro
"""
from choiceModels import MultinomialLogit, NonLinearMLogit
from choiceModels import ChoiceDataset
import torch 
import torch.nn as nn 
from torch.autograd import Variable 
import csv

torch.manual_seed(1) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) 
datapath = "/scratch/users/munro/code/rest_choice.h5" 
featurepath = "/scratch/users/munro/code/variables.tsv" 
#featureCols=["log_distance","rating_overall"]
with open(featurepath) as tsvfile: 
    reader = csv.reader(tsvfile,delimiter='\t') 
    for row in reader: 
        featureCols=row 
choiceVar = "chosen"
train_id = [0,70523] 
dev_id = [70524,74523] 
test_id = [74524,78523] 
train_data = ChoiceDataset(datapath,train_id,512,choiceVar,featureCols)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=False, num_workers=0)

model = NonLinearMLogit(train_data.n_x)
model.to(device)  
#model = MultinomialLogit(train_data.n_x)

#construct loss function and optimizer 
criterion = nn.NLLLoss(reduction='sum') 
#this is batch stochastic gradient descent with momentum 
optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) 

#train model 
for epoch in range(1): 
    corrects = 0 
    running_loss = 0 
    for data in train_loader: 
        inputs, labels,idx = data 
        inputs, labels,idx = Variable(inputs.float()), Variable(labels),Variable(idx)   
        inputs, labels,idx = inputs.to(device), labels.to(device), idx.to(device)  
        optimizer.zero_grad() 
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        loss = criterion(outputs,labels)
        corrects += torch.sum(labels.data == preds)
        running_loss += loss.data.item() 
        loss.backward() 
        #step of SGD 
        optimizer.step()
        #for name, param in model.named_parameters(): 
        #    print(name,param.data)  
    print(epoch," training Loss: ", running_loss/len(train_data))
    print("Training number correct: ", 100*corrects.item()/len(train_data))
    for name, param in model.named_parameters(): 
        if param.requires_grad: 
            print(name,param.data) 

torch.save(model,"nonlinear_mlogit_gpu.torch") 
#model = torch.load(filepath)     



model.eval() 
dev_data = ChoiceDataset(datapath,dev_id,1000,choiceVar,featureCols) 

dev_loader = torch.utils.data.DataLoader(dev_data,batch_size=8,shuffle=False,num_workers=0)
        
corrects=0 
running_loss=0 

for data in dev_loader: 
    inputs, labels, idx = data 
    inputs, labels, idx = Variable(inputs.float()), Variable(labels), Variable(idx) 
    inputs, labels, idx = inputs.to(device), labels.to(device), idx.to(device) 
    devOut = model(inputs) 
    _,preds=torch.max(devOut.data,1) 
    corrects+=torch.sum(labels.data==preds) 
    loss=criterion(devOut,labels)
    running_loss+=loss.data.item() 

print("Dev loss: ", running_loss/len(dev_data)) 
print("Dev correct: ", 100*corrects.item()/len(dev_data))  
#check if parameters correct vs. R 
#print(beta/beta[0])

#print(model(Variable(x[1:5,:,:].data)))
#hour_var = Variable(torch.Tensor([[1.0]]))
##print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
##hour_var = Variable(torch.Tensor([[7.0]]))
#print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)
