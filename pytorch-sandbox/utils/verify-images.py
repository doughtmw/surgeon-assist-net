import os.path
import pandas as pd

# Iterate through CSV and image path location to make sure that
# no images are missing from the data set.

# Define the csv location with all image data listed
img_paths = '../data/csv/img_data.csv' 
data_dir = '../data/data_256/'

# Read in the csv data
all_imgs = pd.read_csv(img_paths, usecols=['Files'], dtype=str)
missing_count = 0

# Iterate through the file list and confirm that
# all images exist in memory
for index, row in all_imgs.iterrows():
   
   # Extract the image name from row
   img_file = row['Files']
   
   # Print if the image file does not exist
   data_path = data_dir + str(img_file) + '.jpg'

   if not os.path.isfile(data_path):
      missing_count += 1
      print("File missing: ", img_file)

print("Done searching for files, there were: ", missing_count, " missing.")