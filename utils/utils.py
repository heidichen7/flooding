import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pickle
import sys
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import const

use_gpu = torch.cuda.is_available()

def imshow(inp, title=None):
    """
    Displays batch of images and corresponding labels.
    Use with show_databatch.

    inp: batch of images
    title: list of labels for each image
    """
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10)) #resize
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    """
    Usage:
    inputs, classes = next(iter(dataloaders[TRAIN]))
    show_databatch(inputs, classes)
    """
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[const.CLASS_NAMES[int(x)] for x in list(classes)])


def visualize_model(vgg, test_data, num_images=6):
    """
    For given trained model, displays sample predicted labels and images.
    """
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in tqdm(enumerate(test_data)):
        inputs, labels = data
        size = inputs.size()[0]

        with torch.no_grad():
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            predicted_labels = [preds[j].item() for j in range(inputs.size()[0])]

            print("Ground truth:")
            show_databatch(inputs.data.cpu(), labels.data.cpu())
            print("Prediction:")
            show_databatch(inputs.data.cpu(), np.clip(predicted_labels, 0, 1))

            del inputs, labels, outputs, preds, predicted_labels
            torch.cuda.empty_cache()

            images_so_far += size
            if images_so_far >= num_images:
                break

    vgg.train(mode=was_training) # Revert model back to original training state

def eval_model(vgg, test_data, criterion):
    """
    Displays model train/test accuracy.
    Example criterion: "nn.CrossEntropyLoss()""
    """
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0


    test_batches = len(test_data)
    print("Evaluating model")
    print('-' * 10)

    print(enumerate(test_data))
    for i, data in tqdm(enumerate(test_data)):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data
        with torch.no_grad():
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss_test += loss.data.item()
            acc_test += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

    avg_loss = loss_test / len(test_data.dataset)
    avg_acc = acc_test.item() / len(test_data.dataset)

#     print (acc_test.cpu() / dataset_sizes[TEST])

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

def train_model(train_data, val_data, vgg, criterion, optimizer, scheduler=None, num_epochs=10, log_freq=2):
    tbx = SummaryWriter('save/')
    startrun = str(datetime.now())
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    train_batches = len(train_data)
#     return None
    val_batches = len(val_data)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in tqdm(enumerate(train_data)):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            # Use half training dataset
            if i >= train_batches / 2:
                break

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
#             inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds, data, loss
            torch.cuda.empty_cache()

        loss_hist.append(loss_train)

        print()
        # * 2 as we only used half of the dataset
        print (acc_train)
        avg_loss = loss_train * 2 / len(train_data.dataset)
        avg_acc = acc_train.item() * 2 / len(train_data.dataset)
        train_acc_hist.append(avg_acc)
        tbx.add_scalar('train/loss', avg_loss, epoch)
        tbx.add_scalar('train/accuracy', avg_acc, epoch)
        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(val_data):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.item()

                acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds, data, loss
            torch.cuda.empty_cache()
        if ((epoch+1) % log_freq == 0):
            torch.save(vgg.state_dict(), 'save/weights_' + str(epoch) + '.pth')


        avg_loss_val = loss_val / len(val_data.dataset)
        avg_acc_val = acc_val.item() / len(val_data.dataset)
        val_acc_hist.append(avg_acc_val)
        tbx.add_scalar('val/loss/' + startrun , avg_loss_val, epoch)
        tbx.add_scalar('val/accuracy/' + startrun , avg_acc_val, epoch)
        print()
        print("Epoch {} result: ".format(epoch + 1))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

        del avg_loss_val, avg_acc_val, avg_loss, avg_acc
        torch.cuda.empty_cache()

    elapsed_time = time.time() - since
    print()

    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    torch.save(best_model_wts, 'save/weights/best.pth')
    vgg.load_state_dict(best_model_wts)
    return vgg, loss_hist, train_acc_hist, val_acc_hist
