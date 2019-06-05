import torch
from dataset import load_data
from utils import eval_model, visualize_model, train_model
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
import torch.nn as nn
import models as flood_models


train_data, val_data, test_data = load_data()
model = ResidualAttentionModel()
#model = flood_models.baselineVGG16()
criterion = nn.CrossEntropyLoss()

wpath = '/home/parsley789/data/weights-attention.pth')
#wpath = '/home/parsley789/data/old_models/vgg16_trained_adam.pt'
model.load_state_dict(torch.load(wpath))
model.cuda()

eval_model(model, test_data, criterion)
#visualize_model(model, test_data)
