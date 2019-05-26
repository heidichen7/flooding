import pandas as pd 
import csv
import glob
import shutil
import os
import os.path
from pathlib import Path
import ntpath

cityscape_src_dir = "data/leftImg8bit"

#writing a cityscapes relevance file
with open('cityscapes.txt', 'w') as file:
	for pngfile in Path(cityscape_src_dir).glob('**/*.png'):
		file.write(ntpath.basename(os.path.splitext(pngfile)[0]) + '\n')

file.close()