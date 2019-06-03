import pandas as pd 
import glob
import shutil
import os
from pathlib import Path

#label images 0 or 1
relabel_data = False
#relevance directory
data_dir = "data/relevance/"
if relabel_data:
	#read in files (lists of image ids for each category - some overlap)
	depth = pd.read_csv(data_dir + "depth.txt", header=None)
	depth[0] = depth[0] + ".jpg"
	flooding = pd.read_csv(data_dir + 'flooding.txt', header=None)
	flooding[0] = flooding[0] + ".jpg"
	irrelevant = pd.read_csv(data_dir + 'irrelevant.txt', header=None)
	irrelevant[0] = irrelevant[0].astype(str) + ".jpg"
	pollution = pd.read_csv(data_dir + 'pollution.txt', header=None)
	pollution[0] = pollution[0] + ".jpg"
	cityscapes = pd.read_csv(data_dir + 'cityscapes.txt', header=None)
	cityscapes[0] = cityscapes[0] + ".png"

	#concatenate images, label 1 if in flooding set
	all_images = pd.concat([flooding, irrelevant, pollution, depth, cityscapes]).drop_duplicates()
	all_images[0] = all_images[0].astype(str)
	all_images['is_flooded'] = all_images.isin(flooding) * 1.0

	#output to csv (no header)
	all_images.to_csv(data_dir + "relevance_data_3.csv", header=None, index=False)

else:
	all_images = {}
	with open(data_dir + "relevance_data_3.csv") as f:
		for line in f:
			key, val = line.split(",")
			all_images[key] = float(val)

	# all_images = pd.read_csv(data_dir + "labeled_relevance.csv", header=None)
	# all_images.columns = ["image_id", "is_flooded"]

#output images into subfolders
# img_src_dir = "data/imgs_small"
img_src_dir = "data/test_data/imgs"
flood_dir = "data/test_data/flood"
not_flood_dir = "data/test_data/not_flood"

for jpgfile in glob.iglob(os.path.join(img_src_dir, "*.jpg")):
	image_id = jpgfile.replace(img_src_dir, '')[1:] #after the slash, on
	if all_images[image_id] == 0:
		shutil.copy(jpgfile, not_flood_dir)
	else:
		shutil.copy(jpgfile, flood_dir)

    # shutil.copy(jpgfile, dst_dir)

#for the cityscape images
cityscape_src_dir = "data/leftImg8bit"
for pngfile in Path(cityscape_src_dir).glob('**/*.png'):
	shutil.copy(pngfile, not_flood_dir)



