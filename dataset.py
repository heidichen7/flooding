from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import const
from PIL import Image


class FloodDataset(Dataset):
    def __init__(self, csv_path, image_path, transforms_dict=None):
        """
        Args:
            csv_path (string): path to csv file. column 0 = image names (with .jpg), column 1 = flood labels
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transforms_dict
        # regardless of transforms, will set output to tensor
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        #save path for get_item
        self.image_path = image_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_path + self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        #Transforms
        if self.transforms is not None:
            img_as_img = self.transforms(img_as_img)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

def load_data(presaved=True):
    """
    returns dataloaders for train, val, and test.
    """
    #add options for splitting / labeling if not already in os
    data_transforms = {
        'train': transforms.Compose([
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally.
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    }

    train_set = FloodDataset(const.TRAIN_CSV_PATH, const.TRAIN_PATH, data_transforms['train'])
    val_set = FloodDataset(const.VAL_CSV_PATH, const.VAL_PATH, data_transforms['val'])
    test_set = FloodDataset(const.TEST_CSV_PATH, const.TEST_PATH, data_transforms['test'])

    train_data, val_data, test_data = [DataLoader \
    (x, batch_size=8,shuffle=True, num_workers=4) for x in [train_set, val_set, test_set]]
    return train_data, val_data, test_data
