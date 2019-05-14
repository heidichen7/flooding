import pandas as pd 

#relevance directory
data_dir = "/data/relevance/"

#read in files (lists of image ids for each category - some overlap)
depth = pd.read_csv(data_dir + "depth.txt", header=None)
flooding = pd.read_csv(data_dir + 'flooding.txt', header=None)
irrelevant = pd.read_csv(data_dir + 'irrelevant.txt', header=None)
pollution = pd.read_csv(data_dir + 'pollution.txt', header=None)

#concatenate images, label 1 if in flooding set
all_images = pd.concat([flooding, irrelevant, pollution, depth]).drop_duplicates()
all_images['is_flooded'] = all_images.isin(flooding) * 1.0

#output to csv (no header)
all_images.to_csv(data_dir + "labeled_relevance.csv", header=None)