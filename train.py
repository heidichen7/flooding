import utils
from dataset import load_data
import models as flood_models

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
import copy

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    loss_hist = []

    train_batches = len(dataloaders[TRAIN])
#     print (train_batches)
#     return None
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
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

            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.data[0]
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        loss_hist.append(loss_train)

        print()
        # * 2 as we only used half of the dataset
        print (acc_train)
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train.item() * 2 / dataset_sizes[TRAIN]

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(dataloaders[VAL]):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = vgg(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss_val += loss.data[0]
                acc_val += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            avg_loss_val = loss_val / dataset_sizes[VAL]
            avg_acc_val = acc_val.item() / dataset_sizes[VAL]

            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()

            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                best_model_wts = copy.deepcopy(vgg.state_dict())

        elapsed_time = time.time() - since
        print()

        print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Best acc: {:.4f}".format(best_acc))

        vgg.load_state_dict(best_model_wts)
        return vgg, loss_hist

def main():
    #data data data!
    train_data, val_data, test_data = load_data()

    #initialize model
    model = flood_models.baseline()
    if use_gpu:
        model.cuda() #.cuda() will move everything to the GPU side

    #define loss, optimizer, loss decay
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #test before training (optional)
    eval_model(model, criterion)
    #train
    trained_model, loss_hist = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

    #evaluate
    eval_model(trained_model, criterion)
    visualize_model(trained_model)

if __name__ == "__main__":
    main()
