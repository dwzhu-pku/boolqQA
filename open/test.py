# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import utils

from sys import platform
from data import LCQMC_Dataset, load_embeddings
from abcnn import ABCNN



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
    accuracy = 0.0
    all_prob = []
    all_labels = []
    
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (q, _, h, _, label) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            q1 = q.to(device)
            q2 = h.to(device)
            labels = label.to(device)

            _, probs = model(q1, q2)
            accuracy += utils.correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probs[:,1].cpu().numpy())
            all_labels.extend(label)

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)


def main(test_file, vocab_file, embeddings_file, pretrained_file, max_length=50, gpu_index=0, batch_size=128):
    
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)

    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)

    print("\t* Loading test data...")    
    test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    print("\t* Building model...")
    model = ABCNN(embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])

    print(20 * "=", " Testing ABCNN model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    main("../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")