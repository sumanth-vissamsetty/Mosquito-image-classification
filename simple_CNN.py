#This script creates, evaluates and saves a simple CNN model for the mosquito image classification task.
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import os, os.path
import tkinter as tk
from tkinter import filedialog
import pandas as pd  


def get_time():
    current_time = ''
    for i in range(6):
        current_time = current_time + str(time.localtime()[i])+'_'
    current_time = current_time[:len(current_time)-1]
    return current_time


def get_input_dir():    
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing data')
    return data_dir


def create_CNN_model(data_dir): 
    # dimensions of our images.
    img_width, img_height = 150, 150
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    #Creating the model.
    model = Sequential()
    #Adding layers to the model.
    model.add(Conv2D(32, (3, 3), input_shape=input_shape)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #Compiling the model.
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', #changed from rmsprop to adam
                  metrics=['accuracy'])
    
    train_data_dir = data_dir+'/train'
    validation_data_dir = data_dir+'/validation'
    #test_data_dir = data_dir+'/test'
    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
    nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
    epochs = 2
    batch_size = 25
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale = 1./255)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255)
                                      #zca_whitening=False)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    
    #print(hist.history)
    model.save('first_model.h5')
    model.save_weights('first_try.h5')
    return hist, model, validation_generator


def evaluate_CNN(hist, model, validation_generator):   
    model.evaluate_generator(generator=validation_generator)
    #Performance metrics
    test_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
    #Predicts the class based on the probability into array of arrays, in which the sub-array is the probability for each test image.
    #Array shape=> [(test_size, 1)]
    predictions = model.predict_generator(validation_generator,steps=test_steps_per_epoch)
    #Reduces the array of arrays to array of size (test_size, 1) and the data type is float
    predicted_class_indices=np.squeeze(predictions,axis=1) 
    #Rounding of the probabilities to get the classes (but note that the classes are in float)
    i=0
    while(i<len(predicted_class_indices)):
        predicted_class_indices[i] = round(predicted_class_indices[i])
        i = i+1
    predictions = predicted_class_indices.astype(int)
    accuracy = cal_acc(hist)
    return predictions, accuracy


def cal_acc(hist):
    #Accuracy metrics
    #f_accuracy.write(hist.history)
    metrics = ['train_accuracy','train_loss','validation_accuracy','validation_loss']
    values = [hist.history['acc'],hist.history['loss'],hist.history['val_acc'],hist.history['val_loss']]
    accuracy = pd.DataFrame({"Metrics":metrics})
    
    #Trasnpose the "values" and add the columns.
    val_t = np.transpose(values)
    i = 0
    for a in val_t:
        accuracy['Epoch-'+str(i)] = a
        i = i+1
    #Trasnpose the final accuracy for desired output.
    accuracy = np.transpose(accuracy)
    return accuracy


def save_results(accuracy, validation_generator, predictions, results_path):
    #Make directory for results
    current_time = get_time()
    training_results = results_path+'training_results'+"_"+current_time
    os.makedirs(training_results)
    
    #Accuracy metrics
    f_accuracy = open(training_results + "\\"+"accuracy"+".csv","w")
    accuracy.to_csv(f_accuracy, index=False)
    f_accuracy.close()
    
    #Confusion matrix
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    matrix = confusion_matrix(true_classes, predictions)
    #print(matrix)
    f_confusion_matrix = open(training_results + "\\"+"confusion-matrix"+".csv",'w')
    f_confusion_matrix.write(np.array2string(matrix, separator=', '))
    f_confusion_matrix.close()
    
    #Results
    result = classification_report(true_classes, predictions, target_names=class_labels)
    #print(result) 
    f_report = open(training_results + "\\"+"report"+".rtf","w")
    f_report.write(result)
    f_report.close()


def main():
    #Processing current_time starts.
    startTime = time.clock()
    print("Process started")
    print("---------------")
    results_path = ('C:\\Users\\svissa1\\Documents\\research\\results\\training\\')
    #Gets the input directory.
    data_dir = get_input_dir()
    #Creates the model and saves it.
    hist, model, validation_generator = create_CNN_model(data_dir)
    #Evaluates the created model.
    predictions, accuracy = evaluate_CNN(hist, model, validation_generator)
    #Saves the results in appropriate files.
    save_results(accuracy, validation_generator, predictions, results_path)

    endtime = time.clock()
    timeElapsed = endtime - startTime
    print("Done creating and saving the model and weights :) in "+str(timeElapsed)+" seconds (="+str(timeElapsed/60)+ " minutes, " +str(timeElapsed/3600)+ " hours, "+str(timeElapsed/86400)+" days)\n")


if __name__ == "__main__":
    main()