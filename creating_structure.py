import os, shutil, os.path
import tkinter as tk
from tkinter import filedialog
import random

root = tk.Tk()
root.withdraw()

input_data_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing input data')
training_data_size = 2000
train_data_percent = 0.7 #=> out of "training_data_size" images 70% for train and 30% for validation and rest for test.

# The parent directory should not be in a directory that has another folder named data
# This will allow you to choose the "parent" director that contains the folders with the different classes
# No other files or folders in that parent directory permitted  
input_folders = os.listdir(input_data_dir)
#parent_dir = os.path.dirname(input_data_dir)
#Folder where the data is split into train, validation and test
data_split_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory to save the data')
data_split_dir = data_split_dir+'/data'
train_data = data_split_dir+'/train'
validation_data = data_split_dir+'/validation'
test_data = data_split_dir+'/test'

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
        os.makedirs(test_data+'/'+folder)

for folder in input_folders:
    if(folder != 'desktop.ini'):
       current_input_folder = input_data_dir+'/'+folder
       #current_output_folder = train_data+'/'+folder
       number_of_images = sum([len(files) for r, d, files in os.walk(current_input_folder)])
       print(number_of_images)
       if(folder == 'Pupae'):
           train_size = train_data_percent * training_data_size * 0.65
           valid_size = 0.3 * training_data_size * 0.65 + train_size
       else:
           train_size = train_data_percent * training_data_size * 0.35
           valid_size = 0.3 * training_data_size * 0.35 + train_size
       train_folder = train_data+'/'+folder+'/'
       validation_folder = validation_data+'/'+folder+'/'
       test_folder = test_data+'/'+folder+'/'
       i=0
       #Retreving the filenames of the images.
       filenames = os.listdir(current_input_folder)
       filenames = [os.path.join('', f) for f in filenames if f.endswith('.jpg')]
       #Sorting the filenames.
       filenames.sort()
       #Randomizing the order of the filenames.
       random.seed(random.randint(1,1000))
       random.shuffle(filenames)
       for image in filenames:#os.listdir(current_input_folder):
           im = current_input_folder+'/'+image
           print(im)
           #os.chmod(im, stat.S_IWOTH)
           if (i < train_size):
               shutil.copy(im, train_folder)
           elif(i > train_size) and (i < valid_size):
               shutil.copy(im, validation_folder)
           elif (i > valid_size) and (i < number_of_images):
               shutil.copy(im, test_folder)    
           i = i+1
print("Done :) with creating the structure and copying images")
