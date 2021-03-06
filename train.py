from utils import eval_model, visualize_model, train_model
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
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

def main(modeltype='attention'):
    #data data data!
    train_data, val_data, test_data = load_data()

    #initialize model
    if modeltype == 'vgg':
        model = flood_models.baselineVGG16()
    elif modeltype == 'resnet':
        model = flood_models.baselineResNet()
    elif modeltype == 'attention':
        model = ResidualAttentionModel()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda() #.cuda() will move everything to the GPU side

    #define loss, optimizer, loss decay
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #test before training (optional)
    #eval_model(model, test_data, criterion)
    #train
    trained_model, loss_hist_train, loss_hist_val = train_model( train_data, val_data, model, criterion, optimizer_ft,  num_epochs=100)

    #evaluate
    eval_model(trained_model, test_data, criterion)
    visualize_model(trained_model, test_data)

if __name__ == "__main__":
    main('resnet')
