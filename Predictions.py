import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import shutil

strTime = ''
for i in range(6):
    strTime = strTime + str(time.localtime()[i])+'_'

strTime = strTime[:len(strTime)-1]

print("Process started")# at "+ str(startTime))
print("---------------")

root = tk.Tk()
root.withdraw()

#Updated use file dialogue
results_path = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory to save the results')
#results_path = ('C:\\Users\\svissa1\\Documents\\research\\results\\testing\\')
predicted_results = results_path+'predicted_images'+"_"+strTime
os.makedirs(predicted_results)
predictions_csvfile = open(predicted_results + "\\" + "predictions"+".csv","w")

model_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing model')
data_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing data')
train_data_dir = data_dir+'/train'
test_data_dir = data_dir+'/test'
# If you want to predict on new images put them in the test data. 

#Processing time starts
startTime = time.clock()

batch_size = 25
img_width, img_height = 150, 150

model = load_model(model_dir+'/first_model.h5')
model.load_weights(model_dir+'/first_try.h5')

##Train stuff is only required for getting the labels.
train_datagen = ImageDataGenerator(
        rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory= test_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

##Prediction and Performance metrics
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
#You need to reset the test_generator before whenever you call the predict_generator. 
#This is important, becuase if you forget to reset the test_generator you will get outputs in a weird order.
test_generator.reset()
#Predicts the class based on the probability into array of arrays, in which the sub-array is the probability for each test image.
#Array shape=> [(test_size, 1)]
predictions=model.predict_generator(test_generator,steps=test_steps_per_epoch)
#Reduces the array of arrays to array of size (test_size, 1) and the data type is float
predicted_class_indices=np.squeeze(predictions,axis=1) 
#Rounding of the probabilities to get the classes (but note that the classes are in float)
i=0
while(i<len(predicted_class_indices)):
    predicted_class_indices[i] = round(predicted_class_indices[i])
    i = i+1
predicted_class_indices = predicted_class_indices.astype(int)
#Retreving the class labels to change the 0's and 1's to classes in the next step.
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
#The following command changes 0's and 1's into classes.
predictions = [labels[k] for k in predicted_class_indices]
#creating the result file.
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(predictions_csvfile,index=False)
predictions_csvfile.close()
#Creating the folders for each class under predictions folder.
classes = train_generator.class_indices.keys() #Instar4, Pupae
for folder in classes:
    os.makedirs(predicted_results+'/'+folder)

#Change the following into something more general. 

#Copying the predcited images to resepective labelled-folders.
for a in results.values:
    if a[1] == 'Pupae':
        shutil.copy(test_data_dir + '/' +a[0],predicted_results+'\\Pupae')
    else:
        shutil.copy(test_data_dir + '/' +a[0],predicted_results+'\\Instars4')

endTime = time.clock()
timeElapsed = endTime - startTime
print("Done predciting the test data :) in "+str(timeElapsed)+" seconds (="+str(timeElapsed/60)+ " minutes, " +str(timeElapsed/3600)+ " hours, "+str(timeElapsed/86400)+" days)\n")
