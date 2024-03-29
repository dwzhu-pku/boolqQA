

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTM_ATTN,ABCNN,BIMPM,ESIM
from transformers import *

from torchtext import data,vocab
from torchtext.data import Iterator, BucketIterator
from dataProcess import TEXT_Field,LABEL_Field,LENGTH_Field,Mydataset,Mydataset_for_bert


class classify():
    def __init__(self,pattern,with_title,epoch_num,batch_size,lr,patience,device,GLOVE_PATH = "../datafile/glove.6B.100d.txt"):
        self.pattern = pattern

        self.epoch_num = epoch_num
        self.lr = lr
        self.with_title = with_title
        self.patience = patience
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = None

        self.train_dataset = Mydataset('../datafile/train.jsonl',with_title = with_title,is_test=False)
        self.valid_dataset = Mydataset('../datafile/dev.jsonl',with_title=with_title,is_test= False)
        self.test_dataset = Mydataset('../datafile/test.jsonl',with_title=with_title,is_test = True)
        
        if not os.path.exists("../datafile/.vector_cache"):
            os.mkdir("../datafile/.vector_cache")
        TEXT_Field.build_vocab(self.train_dataset, vectors=vocab.Vectors(GLOVE_PATH))
        self.vocab = TEXT_Field.vocab


        if pattern=='lstm_attn':#batch size 128 1e-3 patience 4
            self.network = LSTM_ATTN(vocab = self.vocab, hidden_size1 = 128, hidden_size2 = 64, output_size = 2, dropout = 0.3,device = device)
        elif pattern=='abcnn':#batch size 32 lr 1e-3 patience 4
            self.network = ABCNN(vocab=self.vocab, num_layer=1, linear_size=300, max_length=300, device = device)
        elif pattern=='bimpm':#batch size 32   lr 1e-3 patience 4
            self.network = BIMPM(vocab=self.vocab, hidden_size=100, num_perspective=20, class_size=2, device=device)
        elif pattern == 'esim':#batch size 128 lr 1e-3 patience 4 
            self.network = ESIM(vocab=self.vocab, hihdden_size=100,dropout=0.5, num_classes=2, device=device)
        

        elif pattern =='bert':#batch size 16 5e-5 patience 2
            self.network =BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 2).to(device)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif pattern =='roberta-base':#batch 16 8e-6 patience2
            self.network = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels = 2).to(device)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif pattern =='roberta-large': #batch 16 5e-6 patience 2
            self.network = RobertaForSequenceClassification.from_pretrained('roberta-large',num_labels = 2).to(device)
            self.tokenizer =  RobertaTokenizer.from_pretrained('roberta-large')
            
        
        

        self.train_dataset_for_bert = Mydataset_for_bert('../datafile/train.jsonl',self.tokenizer,with_title = with_title,is_test=False)
        self.valid_dataset_for_bert = Mydataset_for_bert('../datafile/dev.jsonl',self.tokenizer,with_title = with_title,is_test=False)
        self.test_dataset_for_bert = Mydataset_for_bert('../datafile/test.jsonl',self.tokenizer,with_title = with_title,is_test=True)

    def train(self):
        s = time.time()
        print('start training,lr=',self.lr)

        #用来保存目前的最大值
        accu = 0

        #choose the iter by pattern
        train_iter = None

        if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
            train_iter = BucketIterator(
                self.train_dataset,train = True,batch_size = self.batch_size,device=self.device,sort_within_batch=False,sort = False,repeat=False
            )
        else:
            train_iter = DataLoader(dataset=self.train_dataset_for_bert,batch_size=self.batch_size,shuffle=True)
    
        #choose the optimizer towards related network,利用filter去过滤掉requires_grad为false的
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.network.parameters()),lr = self.lr)
        #decide how the optimizer updates
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.7, patience=self.patience)

        criterion =nn.CrossEntropyLoss()
        

        for epoch in range(self.epoch_num):
            self.network.train()
            loss_sum = 0

            tqdm_iterator = tqdm(train_iter)
            for iteration,batch_data in enumerate(tqdm_iterator):#按照batch给出
                optimizer.zero_grad()
                #torch.cuda.empty_cache()
                #不同pattern选用不同的dataset因此返回的batch_data的格式也不相同
                if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
                    ids_psg = batch_data.passage.to(self.device)
                    ids_qst = batch_data.question.to(self.device)
                    lens_psg = batch_data.len_passage.to(self.device)
                    lens_qst = batch_data.len_question.to(self.device)
                    labels = batch_data.label.to(self.device)
                else:
                    input_ids,attention_mask,labels = batch_data
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                if self.pattern=='lstm_attn':
                    outputs = self.network.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst)
                elif self.pattern=='abcnn':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='bimpm':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='esim':
                    outputs = self.network.forward(q1=ids_qst,q1_lengths=lens_qst,q2=ids_psg,q2_lengths = lens_psg)
                else :#bert roberta和roberta-large有相同的输入输出形式
                    outputs = self.network.forward(input_ids=input_ids,attention_mask=attention_mask).logits
                    


                loss=criterion(outputs,labels)
                loss_sum+=loss.item()#这里加item避免重复梯度下降
                loss.backward()
                optimizer.step()

            print("lr:",optimizer.state_dict()['param_groups'][0]['lr'],'epoch',epoch,'finished 耗时:',(time.time()-s)/60,'min. loss：',loss_sum)
            print('result for valid:')
            epoch_accu = self.eval(need_load=False)

            scheduler.step(epoch_accu)
            
            if epoch_accu > accu:
                self.save_parameter()#已保存
                accu=epoch_accu
            
    def eval(self,need_load):
        if need_load==True:#如果需要，则load
            self.load_parameter()
        self.network.eval()
        valid_iter = None
        
        #close
        if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
            valid_iter=Iterator(
                self.valid_dataset,train = False,batch_size = self.batch_size,device=self.device,sort_within_batch=False,sort = False,repeat=False
            )
        #open
        else:
            valid_iter = DataLoader(dataset=self.valid_dataset_for_bert,batch_size=self.batch_size)
        TP,TN,FP,FN = 0,0,0,0


        tqdm_iterator = tqdm(valid_iter)
        with torch.no_grad():
            for iteration,batch_data in enumerate(tqdm_iterator):#按照batch给出
                
               #不同pattern选用不同的dataset因此返回的batch_data的格式也不相同
                if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
                    ids_psg = batch_data.passage.to(self.device)
                    ids_qst = batch_data.question.to(self.device)
                    lens_psg = batch_data.len_passage.to(self.device)
                    lens_qst = batch_data.len_question.to(self.device)
                    labels = batch_data.label.to(self.device)
                else:
                    input_ids,attention_mask,labels = batch_data
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                if self.pattern=='lstm_attn':
                    outputs = self.network.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst)
                elif self.pattern=='abcnn':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='bimpm':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='esim':
                    outputs = self.network.forward(q1=ids_qst,q1_lengths=lens_qst,q2=ids_psg,q2_lengths = lens_psg)
                else :#bert roberta和roberta-large有相同的输入输出形式
                    outputs = self.network.forward(input_ids=input_ids,attention_mask=attention_mask).logits


                #根据对应规模的batch 累加计算
                batch_size = labels.size(0)
                ones = torch.ones(batch_size,dtype=torch.long).to(self.device)
                zeros = torch.zeros(batch_size,dtype=torch.long).to(self.device)#必须是long类型的才可以

                pred = outputs.argmax(dim=1)
                TP += ((pred==ones)&(labels==ones)).sum()
                TN += ((pred==zeros)&(labels==zeros)).sum()
                FP += ((pred==zeros)&(labels==ones)).sum()
                FN += ((pred==ones)&(labels==zeros)).sum()
        
        P_yes = float(TP)/float(TP+FP) if (TP + TP)!=0 else 0 
        R_yes = float(TP)/float(TP+FN) if (TP + FN)!=0 else 0
        F1_yes = 2 *P_yes * R_yes/(P_yes+R_yes) if (R_yes + P_yes)!= 0 else 0

        P_no = float(TN)/float(TN+FN) if (TN+FN)!=0 else 0
        R_no = float(TN)/float(TN+FP) if (TN+FP)!=0 else 0
        F1_no = 2 * P_no *R_no /(P_no + R_no) if (P_no + R_no)!=0 else 0

        accu = float((TP+TN))/float((TP+TN+FP+FN))
        print('P_yes: ',P_yes,' R_yes: ',R_yes,' F1_yes:',F1_yes,' ---- P_no: ',P_no,' R_no: ',R_no,' F1_no:',F1_no,'---- accu: ',accu)
        return accu

    def vote_eval(self):
        valid_iter=Iterator(
                self.valid_dataset,train = False,batch_size = 32 ,device=self.device,sort_within_batch=False,sort = False,repeat=False
            )
        tqdm_iterator = tqdm(valid_iter)
        TP,TN,FP,FN = 0,0,0,0
        path = '_title_parameter.pth' if self.with_title else '_parameter.pth'

        lstm_attn = LSTM_ATTN(vocab = self.vocab, hidden_size1 = 128, hidden_size2 = 64, output_size = 2, dropout = 0.3,device = self.device)
        filename = '/home/wzr/hw3/parameter/lstm_attn'+path
        lstm_attn.load_state_dict(torch.load(filename))
        lstm_attn.eval()

        abcnn = ABCNN(vocab=self.vocab, num_layer=1, linear_size=300, max_length=300, device = self.device)
        filename = '/home/wzr/hw3/parameter/abcnn'+path
        abcnn.load_state_dict(torch.load(filename))
        abcnn.eval()

        bimpm = BIMPM(vocab=self.vocab, hidden_size=100, num_perspective=20, class_size=2, device=self.device)
        filename = '/home/wzr/hw3/parameter/bimpm'+path
        bimpm.load_state_dict(torch.load(filename))
        bimpm.eval()

        with torch.no_grad():
            for iteration,batch_data in enumerate(tqdm_iterator):#按照batch给出
                ids_psg = batch_data.passage.to(self.device)
                ids_qst = batch_data.question.to(self.device)
                lens_psg = batch_data.len_passage.to(self.device)
                lens_qst = batch_data.len_question.to(self.device)
                labels = batch_data.label.to(self.device)

                outputs = []

                outputs.append(lstm_attn.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst))
                outputs.append (abcnn.forward(q1=ids_qst,q2=ids_psg))
                outputs.append(bimpm.forward(q1=ids_qst,q2=ids_psg))

                batch_size = labels.size(0)
                ones = torch.ones(batch_size,dtype=torch.long).to(self.device)
                zeros = torch.zeros(batch_size,dtype=torch.long).to(self.device)#必须是long类型的才可以

                pred0=outputs[0].argmax(dim=1)
                pred1 = outputs[1].argmax(dim=1)
                pred2 = outputs[2].argmax(dim=1)
                for i in range(3):
                    outputs[i]= outputs[i].argmax(dim=1)
                pred = pred0 + pred1 + pred2
                
                pred = pred > 1#这一步是element wise的操作 十分简洁大于1表示有至少两个选择为yes，反之为no

                TP += ((pred==ones)&(labels==ones)).sum()
                TN += ((pred==zeros)&(labels==zeros)).sum()
                FP += ((pred==zeros)&(labels==ones)).sum()
                FN += ((pred==ones)&(labels==zeros)).sum()
        
        P_yes = float(TP)/float(TP+FP) if (TP + TP)!=0 else 0 
        R_yes = float(TP)/float(TP+FN) if (TP + FN)!=0 else 0
        F1_yes = 2 *P_yes * R_yes/(P_yes+R_yes) if (R_yes + P_yes)!= 0 else 0

        P_no = float(TN)/float(TN+FN) if (TN+FN)!=0 else 0
        R_no = float(TN)/float(TN+FP) if (TN+FP)!=0 else 0
        F1_no = 2 * P_no *R_no /(P_no + R_no) if (P_no + R_no)!=0 else 0

        accu = float((TP+TN))/float((TP+TN+FP+FN))
        print('P_yes: ',P_yes,' R_yes: ',R_yes,' F1_yes:',F1_yes,' ---- P_no: ',P_no,' R_no: ',R_no,' F1_no:',F1_no,'---- accu: ',accu)
        return accu

    def vote_inference(self):
        test_iter=Iterator(
                self.test_dataset,train = False,batch_size = 32 ,device=self.device,sort_within_batch=False,sort = False,repeat=False
            )
        tqdm_iterator = tqdm(test_iter)
        path = '_title_parameter.pth' if self.with_title else '_parameter.pth'

        lstm_attn = LSTM_ATTN(vocab = self.vocab, hidden_size1 = 128, hidden_size2 = 64, output_size = 2, dropout = 0.3,device = self.device)
        filename = '/home/wzr/hw3/parameter/lstm_attn'+path
        lstm_attn.load_state_dict(torch.load(filename))
        lstm_attn.eval()

        abcnn = ABCNN(vocab=self.vocab, num_layer=1, linear_size=300, max_length=300, device = self.device)
        filename = '/home/wzr/hw3/parameter/abcnn'+path
        abcnn.load_state_dict(torch.load(filename))
        abcnn.eval()

        bimpm = BIMPM(vocab=self.vocab, hidden_size=100, num_perspective=20, class_size=2, device=self.device)
        filename = '/home/wzr/hw3/parameter/bimpm'+path
        bimpm.load_state_dict(torch.load(filename))
        bimpm.eval()



        result = []
        with torch.no_grad():
            for iteration,batch_data in enumerate(tqdm_iterator):#按照batch给出
                ids_psg = batch_data.passage.to(self.device)
                ids_qst = batch_data.question.to(self.device)
                lens_psg = batch_data.len_passage.to(self.device)
                lens_qst = batch_data.len_question.to(self.device)

                outputs = []

                outputs.append(lstm_attn.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst))
                outputs.append (abcnn.forward(q1=ids_qst,q2=ids_psg))
                outputs.append(bimpm.forward(q1=ids_qst,q2=ids_psg))

                pred0 = outputs[0].argmax(dim=1)
                pred1 = outputs[1].argmax(dim=1)
                pred2 = outputs[2].argmax(dim=1)
                for i in range(3):
                    outputs[i]= outputs[i].argmax(dim=1)
                pred = pred0 + pred1 + pred2
                
                pred = pred > 1#这一步是element wise的操作 十分简洁大于1表示有至少两个选择为yes，反之为no

                #此时的结果是batch的
                result+=pred.cpu().numpy().tolist()
        
        result = [str(i)+'\n' for i in result]
        print(len(result))
        with open('inference.txt',encoding='utf-8',mode='w+') as f:
            f.writelines(result)

    def inference(self):

        #依据self.pattern选择
        self.load_parameter()
        self.network.eval()
        test_iter = None
        
        #close
        if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
            test_iter=Iterator(
                self.test_dataset,train = False,batch_size = self.batch_size,device=self.device,sort_within_batch=False,sort = False,repeat=False
            )
        #open
        else:
            test_iter = DataLoader(dataset=self.test_dataset_for_bert,batch_size=self.batch_size)

        tqdm_iterator = tqdm(test_iter)
        result= [] 
        with torch.no_grad():
            for iteration,batch_data in enumerate(tqdm_iterator):#按照batch给出
                
               #不同pattern选用不同的dataset因此返回的batch_data的格式也不相同
                if self.pattern == "lstm_attn" or self.pattern=='abcnn' or self.pattern=='bimpm' or self.pattern=='esim':#close
                    ids_psg = batch_data.passage.to(self.device)
                    ids_qst = batch_data.question.to(self.device)
                    lens_psg = batch_data.len_passage.to(self.device)
                    lens_qst = batch_data.len_question.to(self.device)
                else:
                    input_ids,attention_mask,labels = batch_data
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)

                if self.pattern=='lstm_attn':
                    outputs = self.network.forward(ids_psg = ids_psg,ids_qst=ids_qst,lens_psg=lens_psg,lens_qst=lens_qst)
                elif self.pattern=='abcnn':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='bimpm':
                    outputs = self.network.forward(q1=ids_qst,q2=ids_psg)
                elif self.pattern=='esim':
                    outputs = self.network.forward(q1=ids_qst,q1_lengths=lens_qst,q2=ids_psg,q2_lengths = lens_psg)
                else :#bert roberta和roberta-large有相同的输入输出形式
                    outputs = self.network.forward(input_ids=input_ids,attention_mask=attention_mask).logits


                pred = outputs.argmax(dim=1)#此时的结果是batch的
                result+=pred.cpu().numpy().tolist()
        result = [str(i)+'\n' for i in result]
        with open('inference.txt',encoding='utf-8',mode='w+') as f:
            f.writelines(result)

    def save_parameter(self):
        filename = '/home/wzr/hw3/parameter/'+self.pattern+'_parameter.pth'
        if self.with_title:
            filename = '/home/wzr/hw3/parameter/'+self.pattern+'_title_parameter.pth'
        
        torch.save(self.network.state_dict(),filename)

    def load_parameter(self):
        filename = '/home/wzr/hw3/parameter/'+self.pattern+'_parameter.pth'
        if self.with_title:
            filename = '/home/wzr/hw3/parameter/'+self.pattern+'_title_parameter.pth'
        self.network.load_state_dict(torch.load(filename))
