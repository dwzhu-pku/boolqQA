

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LSTM_ATTN
from torchtext import data,vocab
from torchtext.data import Iterator, BucketIterator
from dataProcess import TEXT_Field,LABEL_Field,LENGTH_Field,construct_dataset,Mydataset

class classify():
    def __init__(self,batch_size,epoch_num,lr,device,GLOVE_PATH = "../data/glove.6B.100d.txt"):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.device = device

        self.train_dataset = Mydataset('../data/train.jsonl',False)
        self.valid_dataset = Mydataset('../data/dev.jsonl',False)
        
        if not os.path.exists("../data/.vector_cache"):
            os.mkdir("../data/.vector_cache")
        TEXT_Field.build_vocab(self.train_dataset, vectors=vocab.Vectors(GLOVE_PATH))
        self.vocab = TEXT_Field.vocab
        self.lstm_attn = LSTM_ATTN(
            vocab = self.vocab, requires_gard = True, hidden_size1 = 128, hidden_size2 = 64, output_size = 2, dropout = 0.3, batch_size = self.batch_size, device = device
            )
        self.criterion =nn.CrossEntropyLoss()


    def train(self,model):
        s = time.time()
        print('start training,lr=',self.lr)
        accu = 0#用来保存目前的最大值
        train_iter = BucketIterator(
            self.train_dataset,train = True,batch_size = 128,device=self.device,sort_within_batch=False,sort = False,repeat=False
        )

        optimizer = None
        if model=='lstm_attn':
            optimizer = torch.optim.Adam(self.lstm_attn.parameters(),lr=self.lr)

        for epoch in range(self.epoch_num):
            loss_sum = 0
            for iteration,batch_data in enumerate(train_iter):#按照batch给出
                optimizer.zero_grad()
                ids_psg = batch_data.passage.to(self.device)
                ids_qst = batch_data.question.to(self.device)
                lens_psg = batch_data.len_passage.to(self.device)
                lens_qst = batch_data.len_question.to(self.device)
                labels = batch_data.label.to(self.device)

                if model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst,is_train=True)

                loss=self.criterion(outputs,labels)
                loss_sum+=loss.item()#这里加item避免重复梯度下降
                loss.backward()
                optimizer.step()

                
            
            print('epoch',epoch,'finished 耗时:',(time.time()-s)/60,'min. loss：',loss_sum)

            print('result for valid:')
            temp_result = self.eval(model=model,need_load=False)
            
            if temp_result>accu:
                self.save_parameter(model)#已保存
                accu=temp_result
            
    

    def eval(self,model,need_load):
        if need_load==True:#如果需要，则load
            self.load_parameter(model=model)
        valid_iter=Iterator(
            self.valid_dataset,train = False,batch_size = 128,device=self.device,sort_within_batch=False,sort = False,repeat=False
        )
        
        TP,TN,FP,FN = 0,0,0,0
        with torch.no_grad():
            for iteration,batch_data in enumerate(valid_iter):#按照batch给出
                ids_psg = batch_data.passage.to(self.device)
                ids_qst = batch_data.question.to(self.device)
                lens_psg = batch_data.len_passage.to(self.device)
                lens_qst = batch_data.len_question.to(self.device)
                labels = batch_data.label.to(self.device)
                batch_size = ids_psg.size(0)
                ones = torch.ones(batch_size,dtype=torch.long).to(self.device)
                zeros = torch.zeros(batch_size,dtype=torch.long).to(self.device)#必须是long类型的才可以
            
                if model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst,is_train= False)

                pred = outputs.argmax(dim=1)
                TP += ((pred==ones)&(labels==ones)).sum()
                TN += ((pred==zeros)&(labels==zeros)).sum()
                FP += ((pred==zeros)&(labels==ones)).sum()
                FN += ((pred==ones)&(labels==zeros)).sum()
        
        P = float(TP)/float((TP+FP)) 
        R = float(TP)/float((TP+FN))
        F1 = 2 *P * R/(P+R)
        accu = float((TP+TN))/float((TP+TN+FP+FN))
        print('P: ',P,' R: ',R,' F1:',F1,' accu: ',accu)
        return accu

    def save_parameter(self,model):
        filename = '/home/wzr/hw3/parameter/'+model+'_parameter.pth'
        if model=='lstm_attn':
            torch.save(self.lstm_attn.state_dict(),filename)


    def load_parameter(self,model):
        filename = '/home/wzr/hw3/parameter/'+model+'_parameter.pth'
        if model=='lstm_attn':
            self.lstm_attn.load_state_dict(torch.load(filename))
    
    """
    def inference(self,model):
        self.load_parameter(model=model)
        data_loader = DataLoader(self.test_dataset,batch_size=100)#可以整除
        result = []
        with torch.no_grad():
            for iteration,(ids,label,length) in enumerate (data_loader):
                ids = torch.LongTensor(ids).to(self.device)
                label = torch.LongTensor(label).to(self.device)
                length = torch.LongTensor(length).to(self.device)
            
                if model=='lstm_attn':
                    outputs = self.lstm_attn.forward(ids=ids,lengths = length,is_train=False)

                pred = outputs.argmax(dim=1)#此时的结果是batch的
                result+=pred.cpu().numpy().tolist()
        result = [str(i)+'\n' for i in result]
        with open('inference.txt',encoding='utf-8',mode='w+') as f:
            f.writelines(result)
    """