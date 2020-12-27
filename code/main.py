
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from classifier import classify



if __name__ == "__main__":

    config = classify(
        epoch_num=20,lr=1e-4,
        device= torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
        GLOVE_PATH = "../datafile/glove.6B.100d.txt"
    )#设置device,在nn.module层面设置to(self.device))
    model = 'bert'
    config.train(model=model)
    config.eval(model=model,need_load=False)
    config.eval(model=model,need_load=True)
    
