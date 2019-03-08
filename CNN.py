#This script creates, evaluates and saves a simple CNN model for the mosquito image classification task.
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import os, os.path
import tkinter as tk
from tkinter import filedialog
import pandas as pd  
import matplotlib.pyplot as plt 


def get_input_dir():    
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick the directory containing data')
    return data_dir


def create_CNN_model(data_dir,first_layer_nodes,second_layer_nodes,third_layer_nodes,if_third_layer,opt,results_path): 
    # dimensions of our images.
    img_width, img_height = 200, 200
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    #Creating the model.
    model = Sequential()
    
    #Adding layers to the model.
    model.add(Conv2D(first_layer_nodes, (3, 3), input_shape=input_shape)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(second_layer_nodes, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if if_third_layer == 'T':     
        model.add(Conv2D(third_layer_nodes, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) 
    #Compiling the model.
    #sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy',
                  #optimizer=sgd,
                  optimizer=opt, #changed from rmsprop to adam
                  metrics=['accuracy'])
    
    train_data_dir = data_dir+'/train'
    validation_data_dir = data_dir+'/validation'
    #test_data_dir = data_dir+'/test'
    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
    nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
    epochs = 10
    batch_size = 25
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale = 1./255)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
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
    accuracy = cal_acc(hist)
    model.summary()
    #You can use model.save(filepath) to save a Keras model at a particular location.
    model.save(results_path+'model.h5')
    model.save_weights(results_path+'weights.h5')
    return hist, model, validation_generator, epochs, accuracy, data_dir, batch_size, img_height, img_width


def evaluate_CNN(model, data_dir, batch_size, img_height, img_width):   
    print("Evaluating")
    test_data_dir = data_dir+'/test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    #print(model.evaluate_generator(generator=test_generator))
    loss, acc = model.evaluate_generator(generator=test_generator)
    print("loss:",str(loss))
    print("acc:",str(acc))
    return loss, acc, test_generator


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


def plot_acc(results_path, hist, epochs):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0, 1.05)
    plt.xlim(0, epochs-1)
    plt.legend(['train', 'validation'], loc='upper left')
    #It is very important to have savefig() before show().
    plt.savefig(results_path+'training.png')
    plt.show()


def save_results(accuracy, test_generator, loss, acc, results_path):
    #Accuracy metrics
    f_accuracy = open(results_path+"training"+".csv","w")
    accuracy.to_csv(f_accuracy, index=False)
    f_accuracy.close()
    
    f_test_accuracy = open(results_path+"test"+".txt","w")
    f_test_accuracy.write("Test loss is: "+str(loss))
    f_test_accuracy.write("\nTest accuracy is: "+str(acc))
    f_test_accuracy.close()


def main():
    #Processing current_time starts.
    startTime = time.clock()
    print("Process started")
    print("---------------")
    first_layer_nodes = 8
    second_layer_nodes = 8
    third_layer_nodes = 1
    if_third_layer = 'F'
    opt = 'rmsprop' 
    #Gets the input directory.
    data_dir = get_input_dir()
    print("data_dir: "+data_dir)
    #Determines the output directory
    results_path = os.path.dirname(data_dir) #parent directory
    #Make directory for results
    results_path = results_path+"\\results\\"
    os.makedirs(results_path)
    #Creates the model and saves it.
    hist, model, validation_generator, epochs, accuracy, data_dir, batch_size, img_height, img_width = create_CNN_model(data_dir,first_layer_nodes,second_layer_nodes,third_layer_nodes,if_third_layer,opt,results_path)
    #Evaluates the created model.
    loss, acc, test_generator = evaluate_CNN(model, data_dir, batch_size, img_height, img_width)
    #Plots the model accuracies.
    plot_acc(results_path, hist, epochs)
    #Saves the results in appropriate files.
    save_results(accuracy, test_generator, loss, acc, results_path)

    endtime = time.clock()
    timeElapsed = endtime - startTime
    print("Done creating and saving the model and weights :) in "+str(timeElapsed)+" seconds (="+str(timeElapsed/60)+ " minutes, " +str(timeElapsed/3600)+ " hours, "+str(timeElapsed/86400)+" days)\n")


if __name__ == "__main__":
    main()
