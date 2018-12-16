from choiceModels import ChoiceDataset
import torch
import torch.nn as nn 
from torch.autograd import Variable
import csv 

datapath="/scratch/users/munro/code/rest_choice.h5" 
featurepath = "/scratch/users/munro/code/variables.tsv" 
#featureCols=["log_distance","rating_overall"] 
choiceVar="chosen" 

test_id=[74524,78523] 

with open(featurepath) as tsvfile: 
    reader = csv.reader(tsvfile,delimiter='\t') 
    for row in reader: 
        featureCols = row 

test_data = ChoiceDataset(datapath,test_id,1000,choiceVar,featureCols) 
criterion = nn.NLLLoss(reduction='sum') 


test_loader = torch.utils.data.DataLoader(test_data,batch_size=8,shuffle=False,num_workers=0) 
model = torch.load('/home/users/munro/cnn_embed.torch') 

model.eval() 

corrects = 0 
running_loss = 0 

for data in test_loader: 
    inputs, labels, idx = data 
    inputs, labels, idx = Variable(inputs.float()), Variable(labels), Variable(idx) 
    trainOut = model(inputs,idx) 
    _,preds=torch.max(trainOut.data,1) 
    corrects+=torch.sum(labels.data==preds) 
    loss = criterion(trainOut,labels) 
    running_loss+=loss.data.item() 


print("Test loss: ", running_loss/len(test_data)) 
print("Test correct: " , 100*corrects.item()/len(test_data)) 
