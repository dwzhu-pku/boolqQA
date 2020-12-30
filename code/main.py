
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from classifier import classify



if __name__ == "__main__":

    pattern = 'roberta'
    config = classify(
        pattern=pattern,
        epoch_num=20,lr=5e-6,batch_size =16,patience = 2,
        device= torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
        GLOVE_PATH = "../datafile/glove.6B.100d.txt"
    )#设置device,在nn.module层面设置to(self.device))
    config.train()
    config.eval(need_load=False)
    config.eval(need_load=True)
    
