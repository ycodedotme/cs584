#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%% imports
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime as dt
import os
import random

from tensorflow.keras import optimizers
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing import image_dataset_from_directory
import sys
import pickle

#%% global variables 
img_width = 150
img_height = 150
batch_size = 128

#%% load data
def load_train_data():

    train_classes_path = "data/seg_train/seg_train"
    classes_list = os.listdir(train_classes_path)
    classes_list

    classes_list = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    classes_list

    # Exploring Random Images from Random CLasses
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    counter = 0
    for class_name in classes_list:
        class_path = os.path.join(train_classes_path, class_name)

        random_img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, random_img_name)       

        img = cv2.imread(img_path)

        counter += 1
        # We only have 6 class so plotting img for each class
        plt.subplot(2, 3, counter)

        plt.imshow(img)
        plt.xlabel(img.shape[1])
        plt.ylabel(img.shape[0])
        plt.title(class_name)

    # Load Dataset
    train_dataset = image_dataset_from_directory(
        train_classes_path,
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_width, img_height),
        seed=123,
        validation_split=0.2,
        subset="training")

    val_dataset = image_dataset_from_directory(
        train_classes_path,
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_width, img_height),
        seed=123,
        validation_split=0.2,
        subset="validation")

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    AutoTune = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.cache().prefetch(buffer_size=AutoTune)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AutoTune)
    return  train_dataset,val_dataset

def load_test_data():
    test_path = "data/seg_test/seg_test"
    test_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    test_classes
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    test_dataset = image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        class_names=test_classes,
        image_size=(img_height, img_width)).map(lambda x, y: (normalization_layer(x), y))
    return test_dataset

#%% build model
def build_model():
    
    # make sure all test always return same model and initial weights
    if os.path.exists("initial.h5"):
        return load_model("initial.h5")
    num_classes = 6
    # Building a functional model
    inputs = Input(shape=(img_width, img_height, 3))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    classifier = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=classifier)

    model.save("initial.h5")
    return model

#%% train model manually
def train_model(train_dataset,val_dataset, test_dataset, epochs):
    
    loss     = []
    val_loss = []
    acc      = []
    val_acc  = []
    t_loss   = []
    t_acc    = []
    

    class TestSetCheckingCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("evaluating...")
            rs = self.model.evaluate(test_dataset)
            t_loss.append(rs[0])
            t_acc.append(rs[1])
            
    class SaveModelCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save('output/'+ model_name +"_intel_image_epoch_{}.h5".format(len(loss)))
            
    model_name = "_".join([str(i) for e in epochs for i in e])
    
    callbacks = [
        SaveModelCallback(),
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1,
                                 restore_best_weights=True),
        TestSetCheckingCallback(),
    ]

    model = build_model()
    
    for _, epoch in enumerate(epochs):
        
        if epoch[0] == "SGDM":
            opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = epoch[0]
            
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        

        hist = model.fit(train_dataset, batch_size=batch_size, epochs=epoch[1], validation_data=val_dataset,
                   callbacks=callbacks)
        
        loss.extend(hist.history['loss'])
        val_loss.extend(hist.history['val_loss'])
        acc.extend(hist.history['accuracy'])
        val_acc.extend(hist.history['val_accuracy']) 
    
    return model, {"loss": loss, 
                   "val_loss": val_loss, 
                   "acc": acc, 
                   "val_acc" :val_acc,
                   "t_loss": t_loss,
                   "t_acc": t_acc}

#%% atuo train model
def auto_train_model(train_dataset,val_dataset, test_dataset, epochs):
    
    loss     = []
    val_loss = []
    acc      = []
    val_acc  = []
    t_loss   = []
    t_acc    = []
    
    rmsprop = build_model()
    rmsprop.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # make the model have save weights at begaining
    # rmsprop.save("temp.h5")
    adam    = build_model()
    adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
            
    model_name = "auto_"
    class TestSetCheckingCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("evaluating...")
            rs = self.model.evaluate(test_dataset)
            t_loss.append(rs[0])
            t_acc.append(rs[1])
    class SaveModelCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save('output/'+ model_name +"_intel_image_epoch_{}.h5".format(len(loss)))
    
    callbacks = [
        SaveModelCallback(),
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1,
                                 restore_best_weights=True),
        TestSetCheckingCallback(),
    ]
    
    only_adam = False
    
    for i in range(epochs):
        print("Epoch {}/{}".format(i + 1, epochs))
        if not only_adam:
            hist_rmsprob = rmsprop.fit(train_dataset, batch_size=batch_size, epochs=1, validation_data=val_dataset,
                       callbacks=callbacks)
            
            hist_adam = adam.fit(train_dataset, batch_size=batch_size, epochs=1, validation_data=val_dataset)
            if hist_rmsprob.history['loss'] > hist_adam.history['loss']:
                print("Starting only using adam at {}".format(i + 1))
                only_adam = True
        else:
            hist_rmsprob = rmsprop.fit(train_dataset, batch_size=batch_size, epochs=1, validation_data=val_dataset,
                       callbacks=callbacks)
        
        loss.extend(hist_rmsprob.history['loss'])
        val_loss.extend(hist_rmsprob.history['val_loss'])
        acc.extend(hist_rmsprob.history['accuracy'])
        val_acc.extend(hist_rmsprob.history['val_accuracy']) 

    
    return rmsprop, {"loss": loss, 
                   "val_loss": val_loss, 
                   "acc": acc, 
                   "val_acc" :val_acc,
                   "t_loss": t_loss,
                   "t_acc": t_acc}

        



def read_model():
        from os import listdir
        from os.path import isfile, join
        output_path = "output"
        only_models = [f for f in listdir(output_path) if isfile(join(output_path, f)) and _ismodel(f)]
        print("\n==========All avaliable models=============")
        for index , f in enumerate(only_models):
            print("{}: {}".format(index + 1, f))
            
        print("==============================================\n")
        
        choice = -1
        while choice == -1 or (choice not in range(1, len(only_models) + 1)):
            try:
                c = input("Please select a model to evaluate: ")
                if c.upper() == "Q":
                    sys.exit(0)
                choice = int(c)
            except:
                pass
        
        return load_model("{}/{}".format(output_path, only_models[choice-1]))

def read_train_epochs(taskstr):
    tasks = taskstr.strip().split(" ")
    epochs = []
    current_epoch = ['', '']
    for t in tasks:
        if not current_epoch[0]:
            current_epoch[0] = t
        else:
            try:
                current_epoch[1] = int(t)
            except :
                pass
        
        if current_epoch[1]:
            epochs.append(current_epoch)
            current_epoch = ['', '']
    if not epochs:
        return read_train_epochs(
            input("Please configue your training epochs, e.g.  adam 7  RMSprop 13: \n"))
    
    print("Your traning epochs will be:")
    for index, epoch in enumerate(epochs):
        print("Step {}: Using {} algorithm train {} epochs".format(
            index + 1, epoch[0], epoch[1]))
        
    if input("Confirm for training (Y|N):").upper() == "Y":
        return epochs
    else:
        print("Give up traning.")
        return []
    
    
def plot_performance(hist, model_name):
   fig, [loss, acc] = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
   fig.suptitle("{} Loss and Accuracy".format(model_name), fontsize=16)
   
   loss.set_title("Loss")
   loss.plot(hist["loss"], label="Train")
   loss.plot(hist["val_loss"], label="Validation")
   loss.plot(hist["t_loss"], label="Test")
   loss.legend()
   
   acc.set_title("Accuracy")
   acc.plot(hist["acc"], label="Train")
   acc.plot(hist["val_acc"], label="Validation")
   acc.plot(hist["t_acc"], label="Test")
   acc.legend()
   plt.show()
   fig.savefig('output/{}-at-{}.png'.format(model_name,
                                            dt.now().strftime("%d%m%y-%H%M"))) 


def _help():
    print("""
Help:
1. ./cnn_image_classification.py eval  
    for evalate a existing model
2. ./cnn_image_classification.py manual 
    for training and comparing algorthm manually
3. ./cnn_image_classification.py atuo
    for auto train a model with tuned combing algorthms 
      
""")
    sys.exit(0)

def _ismodel(file):
    if file.endswith(".h5"):
        return True
    else:
        return False
           
   
#%%
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        _help()
    if sys.argv[1] == "eval":
        while True:
            model = read_model()
            model.summary()
            test_dataset = load_test_data()
            rs = model.evaluate(test_dataset)
            print("============== Evaluation Result ================")
            print("Loss = {}, Accuracy = {}".format(rs[0], rs[1]))
            if input("Continue (Y|N):").upper() != "Y":
                print("Bye!")
                break
    elif sys.argv[1] == "manual":
        #%% train model
        epochs = read_train_epochs(" ".join(sys.argv[2:]))
        if epochs:
            train_dataset,val_dataset = load_train_data()
            test_dataset = load_test_data()
            model, hist  = train_model(train_dataset,val_dataset, 
                                       test_dataset, epochs)
            
            model.summary()
            model_name = "_".join([str(i) for e in epochs for i in e])
            plot_performance(hist, model_name)
            pickle.dump(hist, open('output/{}-at-{}.hist'.format(model_name,
                                                     dt.now().strftime("%d%m%y-%H%M")), "wb"))
    else:
        
        epochs = -1
        while epochs == -1:
            try:
                c = input("Please enter how many epochs to train ")
                if c.upper() == "Q":
                    sys.exit(0)
                epochs = int(c)
            except:
                pass
        
        train_dataset,val_dataset = load_train_data()
        test_dataset = load_test_data()
        model, hist  = auto_train_model(train_dataset,val_dataset, 
                                   test_dataset, epochs)
        
        model.summary()
        model_name = "auto_trained_model"
        plot_performance(hist, model_name)
        pickle.dump(hist, open('output/{}-at-{}.hist'.format(model_name,
                                                 dt.now().strftime("%d%m%y-%H%M")), "wb"))
    