#import Setup
import glob
from PIL import Image
import random
import os

#path to dataset
path_to_new_extendet_dataset = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/new_extended_dataset/original'

#get all image in file
for filename in glob.glob("/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/airflow/dags/*.jpg"):
    #split filename to get classification
    filename_splited = filename.split('/')
    filename_split = filename_splited[-1].split('.')
    #save image to dataset and right classification
    for foldername in glob.glob(path_to_new_extendet_dataset+'/*'):
        foldername_split = foldername.split('/')
        if filename_split[0] == foldername_split[-1]:
            random_number = str(random.randint(0,10000))
            img = Image.open(filename)
            class_modell = str(filename_splited[-1])
            img = img.save(foldername+'/'+random_number+'_'+class_modell)
    # remove image in origin folder
    os.remove(filename)
