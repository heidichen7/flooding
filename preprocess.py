import torch
from torch.utils import data
import random

DATA_PATH = 'imgs_unprocessed'

def load_train_valid(DATA_PATH, batch_size=64, valid_size=.1):
    transforms = transforms.Compose([
                           transforms.Resize(224),
                           transforms.ToTensor(),
                       ])
    master = datasets.imagefolder(data_path, transform=transforms)
    n_master = len(master_dataset)
    n_test = int(n * 0.2)
    n_train = n - 2 * n_test
    idx = list(range(n_master))
    random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:(n_train + n_test)]
    test_idx = idx[(n_train + n_test):]

    train_set = data.Subset(master, train_idx)
    val_set = data.Subset(master, val_idx)
    test_set = data.Subset(master, test_idx)

    return train_set, val_set, test_set