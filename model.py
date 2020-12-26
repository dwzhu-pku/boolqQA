import torch
import torch.nn as nn
import torch.nn.functional as F
#freeze
class LSTM(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,embedding_list,freeze,hidden_size,output_size,dropout,batch_size,device):
        """
        parameters:全部都是网络相关的超参数 和数据本身无关
        input_size：也就是预训练的词向量的size
        hidden_size：隐藏层size
        output_size：linear层output_size
        num_layers:stack num,默认为2
        dropout：dropout层的参数
        较好的超参数：lr=1e-3
        """
        super(LSTM,self).__init__()
        pretrained = torch.FloatTensor(embedding_list)
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim
        ).from_pretrained(embeddings = pretrained,freeze=freeze).to(device)#导入外部训练好的glovepython
        
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True).to(device)
        self.dropout_layer=nn.Dropout(dropout).to(device)
        self.linear = nn.Linear(in_features=hidden_size,out_features=output_size).to(device)
        
        self.device = device
        self.batch_size = batch_size
    
    #端对端，输入的i就是ids,直接用embedding_layer
    def forward(self,ids,lengths,is_train = False):

        x = self.embedding_layer(ids) #batch ,seq_length, embedding_dim
        if is_train:
            x=self.dropout_layer(x)#对原始特征做dropout
        inputs = nn.utils.rnn.pack_padded_sequence(x,lengths,enforce_sorted=False,batch_first=True)#按照长度打包, batch,max_length,emdedding_dim
        packed_outputs,(hidden,cell) = self.lstm(inputs)#hidden:1, batch, hidden_size
        packed_outputs,_=nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)#packed_outputs:batch_size,seq_len, hidden_size
        hidden = hidden.squeeze(0)#hidden:batch, hidden_size
        hidden = F.relu(self.linear(hidden))#线性变换后非线性激活
        return hidden

#freeze
class LSTM_ATTN(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,embedding_list,freeze,hidden_size1,hidden_size2,output_size,dropout,batch_size,device):
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
        pretrained = torch.FloatTensor(embedding_list)
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim
        ).from_pretrained(embeddings = pretrained,freeze=freeze).to(device)#导入外部训练好的glovepython
        
        self.dropout_layer=nn.Dropout(dropout).to(device)
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size1,batch_first=True,bidirectional=True).to(device)
        self.tanh = nn.Tanh()#非线性激活,规模不变
        self.attn = nn.Parameter(torch.zeros(hidden_size1 * 2)).to(device)#和每个hidden_vector做乘积之后再sotfmax得到权重
        
        self.linear1 = nn.Linear(in_features=hidden_size1 * 4,out_features=hidden_size2).to(device)#attention + hidden_t
        self.linear2 = nn.Linear(in_features=hidden_size2,out_features=output_size).to(device)
        
        self.device = device
        self.batch_size = batch_size

    def forward(self,ids,lengths,is_train=False):
        x = self.embedding_layer(ids) #batch ,seq_length, embedding_dim
        if is_train:
            x=self.dropout_layer(x)#对原始特征做dropout
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
        
        outputs = self.linear1(outputs)
        outputs=self.linear2(outputs)
        
        return outputs

#freeze
class CNN(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,embedding_list,freeze,in_channels,out_channels,kernal_size,output_size,dropout,batch_size,device):
        """
        parameters:全部都是网络相关的超参数 和数据本身无关
        in_channels:输入的feature 维度
        out_channels:cnn输出的长度
        output_size：linear层output_size
        dropout：dropout层的参数
        较好的超参数：lr=1e-3
        """
        super(CNN,self).__init__()
        pretrained = torch.FloatTensor(embedding_list)
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim
        ).from_pretrained(embeddings = pretrained,freeze=freeze).to(device)#导入外部训练好的glovepython
        
        self.kernal_size = [2,3,4]
        #三个卷积层分别是(1, channels=256, kernal_size=(2, 100)),(1, 256, (3, 100)),(1, 256, (4, 100))
        # 这三个卷积层是并行的，同时提取2-gram、3-gram、4-gram特征
        self.convs=[nn.Conv2d(in_channels=1,out_channels=out_channels,kernel_size=(k,embedding_dim)).to(device) for k in self.kernal_size]
        self.linear = nn.Linear(in_features=out_channels * 3,out_features=output_size).to(device)
        self.dropout_layer=nn.Dropout(dropout).to(device)
        self.device = device
        self.batch_size = batch_size

    def conv_and_pool(self,x,conv):
        """
        x:batch,1,seq,embedding_dim
        经过conv之后得到(b, 256, seq_len-1, 1), (b, 256, seq_len-2, 1), (b, 256, seq_len-3, 1)
        然后再squeeze()后对最后一维进行max_pool（所以用的是maxpool1d）
        """
        output = F.relu(conv(x))#batch,256,seq_len-i,1
        output = output.squeeze()#batch,255,seq_len-i
        output = F.max_pool1d(output,output.size(2)).squeeze()#batch,256,这里用avg是因为max可能完全抹去了各自的特征
        return output

    def forward(self,ids,is_train=False):
        x = self.embedding_layer(ids) #batch ,seq_length, embedding_dim
        if is_train:
            x=self.dropout_layer(x)#对原始特征做dropout
        x = x.unsqueeze(dim = 1)
        outputs = torch.cat([self.conv_and_pool(x,conv) for conv in self.convs],dim=1)#batch,out_channels*3
        outputs = torch.sigmoid(self.linear(outputs))
        return outputs

#freeze
class CRNN(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,embedding_list,freeze,out_channels,hidden_size,output_size,dropout,batch_size,device):
        """
        parameters:全部都是网络相关的超参数 和数据本身无关
        input_size：也就是预训练的词向量的size
        hidden_size：隐藏层size
        output_size：linear层output_size
        num_layers:stack num,默认为2
        dropout：dropout层的参数
        较好的超参数：lr=1e-3
        """
        super(CRNN,self).__init__()
        pretrained = torch.FloatTensor(embedding_list)
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim
        ).from_pretrained(embeddings = pretrained,freeze=freeze).to(device)#导入外部训练好的glovepython
        
        #这里捕捉3-gram
        self.conv = nn.Conv1d(in_channels=embedding_dim,out_channels=out_channels,kernel_size=3,padding=True).to(device)

        self.lstm = nn.LSTM(input_size=out_channels,hidden_size=hidden_size,bidirectional=False,batch_first=True).to(device)
        self.dropout_layer=nn.Dropout(dropout).to(device)
        self.linear = nn.Linear(in_features=hidden_size,out_features=output_size).to(device)#这里使用双向的lstm
        
        self.device = device
        self.batch_size = batch_size

    def forward(self,ids,lengths,is_train=False):
        """
        ids:batch*length
        length:batch
        """
        x = self.embedding_layer(ids) #batch ,seq_length, embedding_dim
        if is_train:
            x=self.dropout_layer(x)#对原始特征做dropout
        x=x.transpose(1,2)#batch,embedding_dim,seq_len
        x=self.conv(x)#batch,out_channels,seq_len
        x=x.transpose(1,2)#batch,seq_len,out_channel
        inputs = nn.utils.rnn.pack_padded_sequence(x,lengths,enforce_sorted=False,batch_first=True)#按照长度打包, batch,max_length,out_channels
        packed_outputs,(hidden,cell) = self.lstm(inputs)#hidden:1, batch, hidden_size
        packed_outputs,_=nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)#packed_outputs:batch_size,seq_len, hidden_size
        hidden = hidden.squeeze(0)#hidden:batch, hidden_size
        outputs = torch.sigmoid(self.linear(hidden))#线性变换后非线性激活
        return outputs



