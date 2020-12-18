"""
@Author: Dawei Zhu
@Date: 2020-12-18
@Description: 本模块利用torchtext搭建数据通路
"""
from torchtext import data, datasets, vocab
from torchtext.data import Iterator, BucketIterator
import torch
import os
from nltk import word_tokenize

VOCAB_SIZE = 10000 # 词表大小，是一个超参数，有待调整
GLOVE_PATH = "D:\Data\glove.6B\glove.6B.50d.txt" # 全局变量，指向预训练词向量

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# 以下定义两个Field
TEXT = data.Field(sequential=True, tokenize=word_tokenize,batch_first=True,
                  lower=True, fix_length=None, init_token="<SOS>", eos_token="<EOS>")
LABEL = data.Field(sequential=False, use_vocab=False,batch_first=True)

# 加载验证集和训练集。测试集相关部分有待补充
valid = data.TabularDataset(path="./dev.json", format="json",
                          fields={"question": ("question", TEXT),
                                  "answer": ("labels", LABEL), "title": ("title", TEXT), "passage": ("passage", TEXT)})

train = data.TabularDataset(path="./train.json", format="json",
                          fields={"question": ("question", TEXT),
                                  "answer": ("labels", LABEL), "title": ("title", TEXT), "passage": ("passage", TEXT)})

# 以下几行构建词表
if not os.path.exists(".vector_cache"):
    os.mkdir(".vector_cache")
TEXT.build_vocab(train, vectors=vocab.Vectors(
    GLOVE_PATH), max_size=VOCAB_SIZE)
LABEL.build_vocab(train)

# 以下获取词表并作简单验证
vocab = TEXT.vocab
print(vocab.freqs.most_common(10))
print(len(vocab.stoi))
print(list(vocab.stoi)[:10])
print(list(vocab.stoi)[-10:])

# 将词表写入本地文件中。这一部分不必须
with open("./vocab", 'w', encoding="utf8") as fout:
    for key, val in vocab.stoi.items():
        fout.write("{}\t{}\n".format(str(key), str(val)))

# 获取训练集和验证集的迭代器
train_iter, val_iter=BucketIterator.splits(
    (train, valid),
    batch_sizes=(128, 128),
    device=device,
    sort_key=lambda x: len(x.passage),
    sort_within_batch=False,
    repeat=False
)

# 简单查看几个batch的维度信息
for idx, Iter in enumerate(train_iter):
    print(Iter)
    if idx > 1:
        break


# 简单的sanity check
for idx, batch in enumerate(train_iter):
    ques,passage,title,label = batch.question,batch.passage,batch.title,batch.labels
    if (idx > 2):
        break
    print("question:",ques)
    print("passage:",passage)
    print("title:",title)
    print("label:",label)

    print(" ".join([vocab.itos[num.item()] for num in ques[0]]))
    print(" ".join([vocab.itos[num.item()] for num in passage[0]]))
    print(" ".join([vocab.itos[num.item()] for num in title[0]]))
    print(label[0].item())
