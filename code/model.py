import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel



#----------------------------------------------LSTM_ATTN----------------------------------------------
class LSTM_ATTN(nn.Module):
    def __init__(self,vocab,requires_gard,hidden_size1,hidden_size2,output_size,dropout,device):
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
    
    def forward(self,ids_psg,ids_qst,lens_psg,lens_qst):
        outputs_psg = self.forward_single(ids_psg,lens_psg)
        outputs_qst = self.forward_single(ids_qst,lens_qst)
        outputs = torch.cat((outputs_psg,outputs_qst),dim = 1)#batch,hidden1*8

        outputs = self.dropout_layer(outputs)
        
        outputs = self.linear1(outputs)
        outputs = self.linear2(outputs)

        return outputs 


#-------------------------------------------Bert--------------------------------------------
class BERT(nn.Module):
    def __init__(self,dropout,device):
        super(BERT,self).__init__()
    
        self.dropout_layer = nn.Dropout(dropout).to(device)
        hidden_size = 768 * 2#concat
        self.linear = nn.Linear(hidden_size,2).to(device)

        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    

    def forward_single(self,ids,msk):
        hidden = self.bert(input_ids = ids,attention_mask = msk).last_hidden_state #1得到的是sentence encoding
        hidden =hidden[:,0,:]#返回第一个,作为分类标准
        return hidden

    def forward(self,ids_psg,msk_psg,ids_qst,msk_qst):
        hidden_psg = self.forward_single(ids_psg,msk_psg)
        hidden_qst = self.forward_single(ids_qst,msk_qst)

        inputs = torch.cat((hidden_psg,hidden_qst), dim=1)
        
        inputs = self.dropout_layer(inputs)
        outputs = self.linear(inputs)

        return outputs



#------------------------------------------ABCNN--------------------------------------
class ABCNN(nn.Module):
    
    def __init__(self, vocab, requires_gard,num_layer, linear_size, max_length, device):
        super(ABCNN, self).__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer.weight.requires_grad = requires_gard
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
    def __init__(self, vocab, requires_gard, hidden_size=100, num_perspective=20, class_size=2, device="gpu"):
        super(BIMPM, self).__init__()

        self.hidden_size = hidden_size
        self.l = num_perspective
        # ----- Word Representation Layer -----
        
        self.embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 100)
        weight_matrix = vocab.vectors
        self.embedding_layer.weight.data.copy_(weight_matrix)
        self.embedding_layer.weight.requires_grad = requires_gard
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