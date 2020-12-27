import torch
import torch.nn as nn
import torch.nn.functional as F
#freeze
class LSTM_ATTN(nn.Module):
    def __init__(self,vocab,requires_gard,hidden_size1,hidden_size2,output_size,dropout,batch_size,device):
        """
        parameters:全部都是网络相关的超参数 和数据本身无关
        input_size：也就是预训练的词向量的size
        hidden_size1：第一层隐藏层size，选取128，bi之后是256
        output_size：linear层output_size
        num_layers:stack num,默认为2
        dropout：dropout层的参数
        较好的超参数：lr=1e-3
        """
        super(LSTM_ATTN,self).__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer.weight.requires_grad = requires_gard
        self.embedding_layer = self.embedding_layer.to(device)
        
        self.dropout_layer=nn.Dropout(dropout).to(device)
        self.lstm = nn.LSTM(input_size=100,hidden_size=hidden_size1,batch_first=True,bidirectional=True).to(device)
        self.tanh = nn.Tanh()#非线性激活,规模不变
        self.attn = nn.Parameter(torch.zeros(hidden_size1 * 2)).to(device)#和每个hidden_vector做乘积之后再sotfmax得到权重
        
        self.linear1 = nn.Linear(in_features=hidden_size1 * 8,out_features=hidden_size2).to(device)#attention + hidden_t
        self.linear2 = nn.Linear(in_features=hidden_size2,out_features=output_size).to(device)

        self.device = device

    def forward_single(self,ids,lengths):
        #输入是单个的ids，lengths，输出是对应的outputs（batch * hiddensize1 * 4）

        x = self.embedding_layer(ids) #batch ,seq_length, embedding_dim
        #利用pack操作，可以将对应位置的hidden置为0而不被attention到
        inputs = nn.utils.rnn.pack_padded_sequence(x,lengths,enforce_sorted=False,batch_first=True)#按照长度打包, batch,max_length,emdedding_dim
        packed_outputs,(hidden,cell) = self.lstm(inputs)#hidden:2, batch, hidden_size1,因为要计算attention,所以packed_outputs要保留
        packed_outputs,_=nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)#packed_outputs:batch_size,seq_len, hidden_size*num_directions
        hidden = hidden.transpose(0,1)#batch,2,hidden_size
        batch = hidden.size(0)
        hidden = hidden.reshape(batch,-1)#batch,hidden_size1 * 2
        
        M = self.tanh(packed_outputs)#batch,seq_len,hidden_size1*num_directions
        alpha = torch.matmul(M,self.attn)#在最后一个维度相乘，batch,seq_len,自动做了squeeze工作
        alpha = F.softmax(alpha,dim=1)#softmax归一化#batch,seq_len
        alpha = alpha.unsqueeze(dim=2)#batch,len_seq,1
        outputs = torch.mul(packed_outputs,alpha)#这里不是做矩阵乘法，而是element-wise相乘，batch,seq_len,hidden_size1*num_directions
        outputs = outputs.sum(dim=1).squeeze()#batch,hidden_size1*2
        outputs = torch.cat((outputs,hidden.squeeze()),1)#batch,hidden_size1*4
        outputs = torch.relu(outputs)
        return outputs
    
    def forward(self,ids_psg,ids_qst,lens_psg,lens_qst,is_train):
        outputs_psg = self.forward_single(ids_psg,lens_psg)
        outputs_qst = self.forward_single(ids_qst,lens_qst)
        outputs = torch.cat((outputs_psg,outputs_qst),dim = 1)#batch,hidden1*8

        if is_train:outputs = self.dropout_layer(outputs)
        
        outputs = self.linear1(outputs)
        outputs = self.linear2(outputs)

        return outputs 

