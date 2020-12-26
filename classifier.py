

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import LSTM,CRNN,CNN,LSTM_ATTN
from dataProcess import 

class classify():

    def __init__(self,embedding_dim,batch_size,epoch_num,lr,device):

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.device = device
        self.vocab,self.embedding_list = get_vocab_and_embedding_list()
        self.train_data,self.valid_data,self.test_data = get_data(self.vocab)

        self.train_dataset = Mydataset(self.train_data)
        self.valid_dataset = Mydataset(self.valid_data)
        self.test_dataset = Mydataset(self.test_data)

        self.lstm = LSTM(num_embeddings=len(self.vocab)+1, embedding_dim=self.embedding_dim,embedding_list=self.embedding_list,freeze=True,
        hidden_size=50,output_size=2,dropout=0.3,batch_size=self.batch_size,device=device
        )

        self.cnn = CNN(num_embeddings=len(self.vocab)+1, embedding_dim=self.embedding_dim,embedding_list=self.embedding_list,freeze=True,
        in_channels=self.embedding_dim,out_channels=256,kernal_size=3,output_size=2,dropout=0.3,batch_size=self.batch_size,device=self.device
        )

        self.crnn = CRNN(num_embeddings=len(self.vocab)+1, embedding_dim=self.embedding_dim,embedding_list=self.embedding_list,freeze=True,
        out_channels =256, hidden_size=100,output_size=2,dropout=0.3,batch_size=self.batch_size,device=device
        )

        self.lstm_attn = LSTM_ATTN(num_embeddings=len(self.vocab)+1, embedding_dim=self.embedding_dim,embedding_list=self.embedding_list,freeze=True,
        hidden_size1=128,hidden_size2=64,output_size=2,dropout=0.3,batch_size=self.batch_size,device=self.device
        )

        self.criterion =nn.CrossEntropyLoss()

    def train(self,model):
        s = time.time()
        print('start training,lr=',self.lr)
        accu = 0#用来保存目前的最大值
        train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,drop_last=True,shuffle=True)
        optimizer = None
        if model=='lstm':
            optimizer = torch.optim.Adam(self.lstm.parameters(),lr=self.lr)
        elif model=='cnn':
            optimizer = torch.optim.Adam(self.cnn.parameters(),lr=self.lr)
        elif model=='crnn':
            optimizer = torch.optim.Adam(self.crnn.parameters(),lr=self.lr)
        elif model=='lstm_attn':
            optimizer = torch.optim.Adam(self.lstm_attn.parameters(),lr=self.lr)

        for epoch in range(self.epoch_num):
            loss_sum = 0
            for iteration,(ids,label,length) in enumerate(train_loader):#按照batch给出
                optimizer.zero_grad()
                ids = torch.LongTensor(ids).to(self.device)
                label = label.to(self.device)
                length = length.to(self.device)
                if model=='lstm':
                    outputs = self.lstm.forward(ids=ids,lengths = length,is_train=True)#batch_size , 2
                elif model=='cnn':
                    outputs = self.cnn.forward(ids=ids,is_train=True)
                elif model=='crnn':
                    outputs = self.crnn.forward(ids=ids,lengths = length,is_train=True)
                elif model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids=ids,lengths = length,is_train=True)

                loss=self.criterion(outputs,label)
                loss_sum+=loss.item()#这里加item避免重复梯度下降
                loss.backward()
                optimizer.step()
            
            print('epoch',epoch,'finished 耗时:',(time.time()-s)/60,'min. loss：',loss_sum)
            print('result for train:')
            result_for_train = self.test(self.train_dataset,model=model,need_load=False)
            print('result for valid:')
            temp_result = self.test(self.valid_dataset,model=model,need_load=False)
            
            if temp_result>accu:
                self.save_parameter(model)#已保存
                accu=temp_result
            
    
    def test(self,dataset,model,need_load):
        if need_load==True:#如果需要，则load
            self.load_parameter(model=model)
        data_loader = DataLoader(dataset,batch_size=100)#可以整除
        ones = torch.ones(100,dtype=torch.long).to(self.device)
        zeros = torch.zeros(100,dtype=torch.long).to(self.device)#必须是long类型的才可以
        TP,TN,FP,FN = 0,0,0,0
        with torch.no_grad():
            for iteration,(ids,label,length) in enumerate (data_loader):
                ids = torch.LongTensor(ids).to(self.device)
                label = torch.LongTensor(label).to(self.device)
                length = torch.LongTensor(length).to(self.device)
            
                if model=='lstm':
                    outputs = self.lstm.forward(ids=ids,lengths = length,is_train=False)#batch_size , 2
                elif model=='cnn':
                    outputs = self.cnn.forward(ids=ids,is_train=False)
                elif model=='crnn':
                    outputs = self.crnn.forward(ids=ids,lengths = length,is_train=False)
                elif model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids=ids,lengths = length,is_train=False)

                pred = outputs.argmax(dim=1)
                TP += ((pred==ones)&(label==ones)).sum()
                TN += ((pred==zeros)&(label==zeros)).sum()
                FP += ((pred==zeros)&(label==ones)).sum()
                FN += ((pred==ones)&(label==zeros)).sum()
        
        P = float(TP)/float((TP+FP)) 
        R = float(TP)/float((TP+FN))
        F1 = 2 *P * R/(P+R)
        accu = float((TP+TN))/float((TP+TN+FP+FN))
        print('P: ',P,' R: ',R,' F1:',F1,' accu: ',accu)
        return accu

    def save_parameter(self,model):
        filename = '/home/wzr/proj2/'+model+'_parameter.pth'
        if model=='lstm':
            torch.save(self.lstm.state_dict(),filename)
        elif model=='cnn':
            torch.save(self.cnn.state_dict(),filename)
        elif model=='crnn':
            torch.save(self.crnn.state_dict(),filename)
        elif model=='lstm_attn':
            torch.save(self.lstm_attn.state_dict(),filename)


    def load_parameter(self,model):
        filename = '/home/wzr/proj2/'+model+'_parameter.pth'
        if model=='lstm':
            self.lstm.load_state_dict(torch.load(filename))
        elif model=='cnn':
            self.cnn.load_state_dict(torch.load(filename))
        elif model=='crnn':
            self.crnn.load_state_dict(torch.load(filename))
        elif model=='lstm_attn':
            self.lstm_attn.load_state_dict(torch.load(filename))
    
    def inference(self,model):
        self.load_parameter(model=model)
        data_loader = DataLoader(self.test_dataset,batch_size=100)#可以整除
        result = []
        with torch.no_grad():
            for iteration,(ids,label,length) in enumerate (data_loader):
                ids = torch.LongTensor(ids).to(self.device)
                label = torch.LongTensor(label).to(self.device)
                length = torch.LongTensor(length).to(self.device)
            
                if model=='lstm':
                    outputs = self.lstm.forward(ids=ids,lengths = length,is_train=False)#batch_size , 2
                elif model=='cnn':
                    outputs = self.cnn.forward(ids=ids,is_train=False)
                elif model=='crnn':
                    outputs = self.crnn.forward(ids=ids,lengths = length,is_train=False)
                elif model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids=ids,lengths = length,is_train=False)

                pred = outputs.argmax(dim=1)#此时的结果是batch的
                result+=pred.cpu().numpy().tolist()
        result = [str(i)+'\n' for i in result]
        with open('inference.txt',encoding='utf-8',mode='w+') as f:
            f.writelines(result)
