from torch.utils.data.dataset import Dataset
from torchvision import transforms
import const

class FloodDataset(Dataset):
    def __init__(self, csv_path, transforms_dict=None):
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

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        #Transforms
        if self.transforms is not None:
            data = self.transforms(data)
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

    train_data = FloodDataset(const.TRAIN_PATH, data_transforms['train'])
    val_data = FloodDataset(const.VAL_PATH, data_transforms['val'])
    test_data = FloodDataset(const.TEST_PATH, data_transforms['test'])

    return train_data, val_data, test_data
