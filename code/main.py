
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from classifier import classify



if __name__ == "__main__":

    pattern = 'bimpm'
    print('pattern: ',pattern)

    config = classify(
        pattern=pattern,
        with_title = True,#with title与否，只是简单将title加到passage之前
        epoch_num=30,lr=5e-4,batch_size =32,patience = 4,
        device= torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
        GLOVE_PATH = "../datafile/glove.6B.100d.txt"
    )#设置device,在nn.module层面设置to(self.device))
    config.train()
    config.eval(need_load=False)
    config.eval(need_load=True)
    config.inference()
    
