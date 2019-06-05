import torch
from torchvision import models
import os
import torch.nn as nn
import const
from model import residual_attention_network

def baselineResNet():
    rn18 = models.resnet18(pretrained=True)

    #for param in rn18.parameters():
    #    param.require_grad = False

    num_ftrs = rn18.fc.in_features
    rn18.fc = nn.Linear(num_ftrs, len(const.CLASS_NAMES))
    return rn18

def baselineVGG16():
    # load in vgg16 model
    if False:
        vgg16 = models.vgg16()
        vgg16.load_state_dict(torch.load("C:/Users/heidi/.torch/models/vgg16-397923af.pth"))
    else:
        vgg16 = models.vgg16(pretrained=True)

    # Freeze training for all layers - we transferin!
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, len(const.CLASS_NAMES))]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    return vgg16
