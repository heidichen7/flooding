import pandas as pd
import numpy as np
import os
import glob
import shutil
import torchvision 
from torchvision import datasets, models, transforms
from sklearn.utils import shuffle

#all relevant directories
data_dir = "data/relevance/"

# read the csv
files = pd.read_csv(data_dir + "relevance_data_3.csv", header=None)
files = shuffle(files)
dataset_size = files.shape[0]

# percentage splits for data set
train_p = 0.6
test_p = 0.2
val_p = 0.2

# indices for slicing
test_i = int(dataset_size * train_p)
val_i = test_i + int(dataset_size * test_p)

# separated into 3 dataframes
train = files[0:test_i]
test = files[test_i: val_i]
val = files[val_i:]

# convert the dataframes into lists
train_imgs = train.iloc[:, 0].tolist()
test_imgs = test.iloc[:, 0].tolist()
val_imgs = val.iloc[:, 0].tolist()

# writing into csvs
train.to_csv(data_dir + "train_data.csv", header=None, index=False)
test.to_csv(data_dir + "test_data.csv", header=None, index=False)
val.to_csv(data_dir + "val_data.csv", header=None, index=False)

# sort images into test, train and val folders
flood_dir = "data/test_data/flood"
not_flood_dir = "data/test_data/not_flood"
test_dir = "data/test"
train_dir = "data/train"
val_dir = "data/val"

#extract all jpgs
for jpgfile in glob.iglob(os.path.join(flood_dir, "*.jpg")):
	image_id = jpgfile.replace(flood_dir, '')[1:]
	if (image_id in train_imgs):
		shutil.copy(jpgfile, train_dir)
	elif (image_id in test_imgs):
		shutil.copy(jpgfile, test_dir)
	elif (image_id in val_imgs):
		shutil.copy(jpgfile, val_dir)

#extract all pngs
for pngfile in glob.iglob(os.path.join(flood_dir, "*.png")):
	image_id = pngfile.replace(flood_dir, '')[1:]
	if (image_id in train_imgs):
		shutil.copy(pngfile, train_dir)
	elif (image_id in test_imgs):
		shutil.copy(pngfile, test_dir)
	elif (image_id in val_imgs):
		shutil.copy(pngfile, val_dir)

#extract all jpgs
for jpgfile in glob.iglob(os.path.join(not_flood_dir, "*.jpg")):
	image_id = jpgfile.replace(not_flood_dir, '')[1:]
	if (image_id in train_imgs):
		shutil.copy(jpgfile, train_dir)
	elif (image_id in test_imgs):
		shutil.copy(jpgfile, test_dir)
	elif (image_id in val_imgs):
		shutil.copy(jpgfile, val_dir)

#extract all pngs
for pngfile in glob.iglob(os.path.join(not_flood_dir, "*.png")):
	image_id = pngfile.replace(not_flood_dir, '')[1:]
	if (image_id in train_imgs):
		shutil.copy(pngfile, train_dir)
	elif (image_id in test_imgs):
		shutil.copy(pngfile, test_dir)
	elif (image_id in val_imgs):
		shutil.copy(pngfile, val_dir)