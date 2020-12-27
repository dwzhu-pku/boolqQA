# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import utils

# from data import load_embeddings
from abcnn import ABCNN
from torchtext import data, vocab
from torchtext.data import Iterator, BucketIterator

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../code'))
from dataProcess import TEXT_Field, LABEL_Field, LENGTH_Field, construct_dataset, Mydataset


def train(model, dataloader, optimizer, criterion, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)

    for batch_index, batch_data in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        q1 = batch_data.question.to(device)
        q2 = batch_data.passage.to(device)
        labels = batch_data.label.to(device)

        optimizer.zero_grad()
        logits, probs = model(q1, q2)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += utils.correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}".format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Move input and output data to the GPU if one is used.
            q1 = batch_data.question.to(device)
            q2 = batch_data.passage.to(device)
            labels = batch_data.label.to(device)

            logits, probs = model(q1, q2)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += utils.correct_predictions(probs, labels)
            all_prob.extend(probs[:,1].cpu().numpy())
            all_labels.extend(labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)




# def main(train_file, dev_file, embeddings_file, vocab_file, target_dir,  
def main(train_file, dev_file, embeddings_file, target_dir,  
         max_length=50,
         epochs=50,
         batch_size=256,
         lr=0.0005,
         patience=5,
         max_grad_norm=10.0,
         gpu_index=0,
         checkpoint=None):

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # save checkpoints
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_dataset = Mydataset(train_file, False)
    TEXT_Field.build_vocab(train_dataset, vectors=vocab.Vectors(embeddings_file))
    train_iter = BucketIterator(train_dataset, train=True, batch_size=128, device=device, sort_within_batch=False, sort=False, repeat=False)
    print("\t* Loading validation data...")
    valid_dataset = Mydataset(dev_file, False)
    valid_iter = Iterator(valid_dataset, train=False, batch_size=128, device=device, sort_within_batch=False, sort=False, repeat=False)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = utils.load_embeddings(embeddings_file)
    model = ABCNN(embeddings, device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, valid_iter, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss, (valid_accuracy*100), auc))

    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ABCNN model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        # train
        print("* Training epoch {}:".format(epoch))
        # epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, criterion, epoch, max_grad_norm)
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_iter, optimizer, criterion, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # eval
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc= validate(model, valid_iter, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n".format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_auc))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))

        # 连续 patience 个 epoch 都没有增长, then stop.
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    

if __name__ == "__main__":
    train_file = '../datafile/train.jsonl'
    dev_file = '../datafile/dev.jsonl'
    embeddings_file = "../datafile/glove.6B.100d.txt"
    main(train_file, dev_file, embeddings_file, "./ckpts")