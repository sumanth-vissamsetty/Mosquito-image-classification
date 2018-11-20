import os, shutil, os.path
import tkinter as tk
from tkinter import filedialog
import random

root = tk.Tk()
root.withdraw()

input_data_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing input data')
input_folders = os.listdir(input_data_dir)
parent_dir = os.path.dirname(input_data_dir)
#Folder where the data is split into train, validation and test
data_split_dir = parent_dir+'/data'
train_data = data_split_dir+'/train'
validation_data = data_split_dir+'/validation'
test_data = data_split_dir+'/test/test'
#Creating the 'data' folder and 'train','validation' and 'test' folders under it.
os.makedirs(data_split_dir)
os.makedirs(train_data)
os.makedirs(validation_data)
os.makedirs(test_data)

#Creating the empty folders for filling in the pictures later
for folder in input_folders:
    if(input_folders != 'data'):
        os.makedirs(train_data+'/'+folder)
        os.makedirs(validation_data+'/'+folder)
        #os.makedirs(test_data+'/'+folder)

for folder in input_folders:
    if(folder != 'desktop.ini'):
       current_input_folder = input_data_dir+'/'+folder
       #current_output_folder = train_data+'/'+folder
       number_of_images = sum([len(files) for r, d, files in os.walk(current_input_folder)])
       print(number_of_images)
       train_size = 0.5 * number_of_images
       validation_size = 0.3 * number_of_images + train_size
       train_folder = train_data+'/'+folder+'/'
       validation_folder = validation_data+'/'+folder+'/'
       test_folder = test_data+'/'#+folder+'/'
       i=0
       #Retreving the filenames of the images.
       filenames = os.listdir(current_input_folder)
       filenames = [os.path.join('', f) for f in filenames if f.endswith('.jpg')]
       #Sorting the filenames.
       filenames.sort()
       #Randomizing the order of the filenames.
       random.seed(234)
       random.shuffle(filenames)
       for image in filenames:#os.listdir(current_input_folder):
           im = current_input_folder+'/'+image
           print(im)
           #os.chmod(im, stat.S_IWOTH)
           if (i < train_size):
               shutil.copy(im, train_folder)
           elif(i > train_size) and (i < validation_size):
               shutil.copy(im, validation_folder)
           elif (i > validation_size) and (i < number_of_images):
               shutil.copy(im, test_folder)    
           i = i+1
print("Done :) with creating the structure and copying images")