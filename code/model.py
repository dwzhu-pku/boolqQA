import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import *

from sys import platform




#----------------------------------------------LSTM_ATTN----------------------------------------------
class LSTM_ATTN(nn.Module):
    def __init__(self,vocab,hidden_size1,hidden_size2,output_size,dropout,device):
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
    
    def forward(self,ids_psg,ids_qst,lens_psg,lens_qst):
        outputs_psg = self.forward_single(ids_psg,lens_psg)
        outputs_qst = self.forward_single(ids_qst,lens_qst)
        outputs = torch.cat((outputs_psg,outputs_qst),dim = 1)#batch,hidden1*8

        outputs = self.dropout_layer(outputs)
        
        outputs = self.linear1(outputs)
        outputs = self.linear2(outputs)

        return outputs 





#------------------------------------------ABCNN--------------------------------------
class ABCNN(nn.Module):
    
    def __init__(self, vocab, num_layer, linear_size, max_length, device):
        super(ABCNN, self).__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer = self.embedding_layer.to(device)

        self.linear_size = linear_size
        self.num_layer = num_layer
        self.max_length = max_length
        self.conv = nn.ModuleList([Wide_Conv(max_length,100, device) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(
            nn.Linear(100*(1+self.num_layer)*2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_size, 2),
        ).to(device)

    def forward(self, q1, q2):
        # question & passage的长度都扩充/截取到max_length
        B, L1 = q1.shape    
        B, L2 = q2.shape
        if L1 < self.max_length:
            zeros = torch.zeros((B,self.max_length-L1),dtype=torch.long).to(self.device)
            q1 = torch.cat((q1, zeros), dim=1)
        else: 
            q1 = q1[:, 0: self.max_length]
        
        if L2 < self.max_length:
            zeros = torch.zeros((B,self.max_length-L2),dtype=torch.long).to(self.device)
            q2 = torch.cat((q2, zeros), dim=1)
        else: 
            q2 = q2[:, 0: self.max_length]

        mask1, mask2 = q1.eq(0), q2.eq(0)
        res = [[], []]
        q1_encode = self.embedding_layer(q1)
        q2_encode = self.embedding_layer(q2)
        
        # eg: s1 => res[0]
        # (batch_size, seq_len, dim) => (batch_size, dim)
        # if num_layer == 0
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode
        # batch_size * (dim*(1+num_layer)*2) => batch_size * linear_size
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        sim = self.fc(x)
        return sim

class Wide_Conv(nn.Module):
    def __init__(self, seq_len, embeds_size, device):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size))).to(device)
        nn.init.xavier_normal_(self.W)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1).to(device)
        self.tanh = nn.Tanh().to(device)
        
    def forward(self, sent1, sent2, mask1, mask2):
        '''
        sent1, sent2: batch_size * seq_len * dim
        '''
        # sent1, sent2 = sent1.transpose(0, 1), sent2.transpose(0, 1)
        # => A: batch_size * seq_len * seq_len
        # print("sent1: ", sent1.shape)
        # print("sent2: ", sent2.shape)
        A = match_score(sent1, sent2, mask1, mask2)
        # attn_feature_map1: batch_size * seq_len * dim
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        # x1: batch_size * 2 *seq_len * dim
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2

def match_score(s1, s2, mask1, mask2):
    '''
    s1, s2:  batch_size * seq_len  * dim
    '''
    batch, seq_len1, dim1 = s1.shape
    batch, seq_len2, dim2 = s2.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len2, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len1, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)

def attention_avg_pooling(sent1, sent2, mask1, mask2):
    # A: batch_size * seq_len * seq_len
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


#-------------------------------------------BIMPM----------------------------
class BIMPM(nn.Module):
    def __init__(self, vocab, hidden_size=100, num_perspective=20, class_size=2, device="gpu"):
        super(BIMPM, self).__init__()

        self.hidden_size = hidden_size
        self.l = num_perspective
        # ----- Word Representation Layer -----
        
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer = self.embedding_layer.to(device)
        self.embeds_dim = 100

        self.class_size = class_size
        self.device = device
        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=self.embeds_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        ).to(device)
        # ----- Matching Layer -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(self.l, self.hidden_size)).to(device))
        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 8,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        ).to(device)
        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2).to(device)
        self.pred_fc2 = nn.Linear(self.hidden_size * 2, self.class_size).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.embedding_layer.weight.data[0], -0.1, 0.1)
        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)
        # ----- Matching Layer -----
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)
        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)
        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)
        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=0.1, training=self.training).to(self.device)

    def forward(self, q1, q2):
        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        p_encode = self.embedding_layer(q1)
        h_endoce = self.embedding_layer(q2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p_encode)
        con_h, _ = self.context_LSTM(h_endoce)
        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)
        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.hidden_size, dim=-1)
        # 1. Full-Matching
        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1, self.l)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2, self.l)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1, self.l)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2, self.l)
        # 2. Maxpooling-Matching
        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3, self.l)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4, self.l)
        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)
        # 3. Attentive-Matching
        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)
        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)
        # 4. Max-Attentive-Matching
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)
        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)
        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)
        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)
        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)
        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)
        
        return x
# ----- Matching Layer -----
def mp_matching_func(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l)
    """
    seq_len = v1.size(1)
    # (1, 1, hidden_size, l)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    # (batch, seq_len, hidden_size, l)
    v1 = w * torch.stack([v1] * l, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * l, dim=3)
    else:
        v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * l, dim=3)
    m = F.cosine_similarity(v1, v2, dim=2)
    return m

def mp_matching_func_pairwise(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l, seq_len1, seq_len2)
    """
    # Trick for large memory requirement
    # (1, l, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)
    # (batch, l, seq_len, hidden_size)
    v1, v2 = w * torch.stack([v1] * l, dim=1), w * torch.stack([v2] * l, dim=1)
    # (batch, l, seq_len, hidden_size->1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)
    # (batch, l, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)
    # (batch, seq_len1, seq_len2, l)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)
    return m

def attention(v1, v2):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """
    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm
    return div_with_small_value(a, d)

def div_with_small_value(n, d, eps=1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d



#---------------------------------------ESIM----------------------------------
class ESIM(nn.Module):
    def __init__(self, vocab,hihdden_size,dropout, num_classes, device):
        super(ESIM, self).__init__()
        self.embedding_dim = 100
        self.hidden_size = hihdden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer = self.embedding_layer.to(device)

        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout).to(self.device)
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True).to(self.device)
        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, self.hidden_size), 
                                        nn.ReLU()).to(self.device)
        self.attention = SoftmaxAttention().to(self.device)
        self.second_rnn = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True).to(self.device)
        self.classification = nn.Sequential(nn.Linear(2*4*self.hidden_size, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.hidden_size//2),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size//2, self.num_classes)).to(self.device)
           
    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = get_mask(q1, q1_lengths).to(self.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.device)
        q1_embed = self.embedding_layer(q1)
        q2_embed = self.embedding_layer(q2)
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        # 双向lstm编码
        q1_encoded = self.first_rnn(q1_embed, q1_lengths)
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
        # concat
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        # 映射一下
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        # 再次经过双向RNN
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)
        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
        # 分类
        logits = self.classification(merged)
        return logits

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch

class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias, 
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #for linux
        reordered_outputs = outputs.index_select(0, restoration_idx)
        return reordered_outputs

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """
    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses       




def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask