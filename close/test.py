# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import utils

from sys import platform
from abcnn import ABCNN
from bimpm import BIMPM
from torchtext import data, vocab
from torchtext.data import Iterator, BucketIterator

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../code'))
from dataProcess import TEXT_Field, LABEL_Field, LENGTH_Field, construct_dataset, Mydataset



def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    all_prob = []
    all_pred = []
    tqdm_batch_iterator = tqdm(dataloader)

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            q1 = batch_data.question.to(device)
            q2 = batch_data.passage.to(device)

            _, probs = model(q1, q2)
            batch_time += time.time() - batch_start
            all_prob.extend(probs[:,1].cpu().numpy())
            _, cur_pred = probs.max(dim=1)
            all_pred.extend(cur_pred)

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    return batch_time, total_time, all_prob, all_pred


def main(train_file, test_file, embeddings_file, pretrained_file, output_path, max_length=50, gpu_index=0, batch_size=128, model_name='ABCNN'):
    
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    
    fw = open(output_path, 'w')

    # Retrieving model parameters from checkpoint.
    embeddings = utils.load_embeddings(embeddings_file)

    print("\t* Loading test data...")    
    train_dataset = Mydataset(train_file, False)
    TEXT_Field.build_vocab(train_dataset, vectors=vocab.Vectors(embeddings_file))
    print('test_file: ', test_file)
    test_dataset = Mydataset(test_file, is_test=True)
    test_iter = Iterator(test_dataset, train=False, batch_size=batch_size, device=device, sort_within_batch=False, sort=False, repeat=False)

    print("\t* Building model...")
    if model_name == 'ABCNN':
        model = ABCNN(embeddings, device=device).to(device)
    elif model_name == 'BIMPM':
        model = BIMPM(embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])

    print(20 * "=", " Testing ABCNN model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, all_prob, all_pred = test(model, test_iter)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s\n".format(batch_time, total_time))
    for pred in all_pred:
        fw.write(str(pred) + '\n')
    fw.close()


if __name__ == "__main__":
    train_file = '../datafile/train.jsonl'  # 传入 train_file 是为了建立词表
    test_file = '../datafile/test.jsonl'
    embeddings_file = "../datafile/glove.6B.100d.txt"
    ckpt_pth = './ckpts'
    main(train_file, test_file, embeddings_file, '%s/best.pth.tar' % ckpt_pth, output_path='%s/test_pred.txt' % ckpt_pth, gpu_index=1, model_name='ABCNN')
