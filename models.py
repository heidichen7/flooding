import torch
from torchvision import models
import os
import torch.nn as nn
import const

# class VGG16_Baseline(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred

# def res_attention():
#     net = CaffeNet(const.PROTOFILE)


def baseline():
    # load in vgg16 model
    if 'heidi' in os.getcwd():
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
