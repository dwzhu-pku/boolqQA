
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from classifier import classify



if __name__ == "__main__":

    config = classify(batch_size=128,epoch_num=100,lr=1e-4,
    device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    GLOVE_PATH = "../data/glove.6B.100d.txt"
    )#设置device,在nn.module层面设置to(self.device))
    model = 'lstm_attn'
    config.train(model=model)
    config.eval(model=model,need_load=False)
    config.eval(model=model,need_load=True)
    
