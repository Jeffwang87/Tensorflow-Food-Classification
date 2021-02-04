"""
Food Images Classification
authors: Jianxin Wang

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt

#set you own directory
DIR = "./images"

#def your own image label depending on your data
def label_img(name):
    word_label = name.split('/')[0]
    if word_label == 'bibimbap':
        return np.array([1,0,0])
    elif word_label == 'dumplings':
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])

def load_training_data():
    train_data = []
    file = open("set your directory", "r")
    for img in file.readlines():
        label =label_img(img)
        path = os.path.join(DIR, img[:-1]) + ".jpg"
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((300, 300), Image.ANTIALIAS)
            train_data.append([np.array(img), label])

    shuffle(train_data)
    return train_data

def load_test_data():
    test_data = []
    file = open("set your directory", "r")
    for img in file.readlines():
        label =label_img(img)
        path = os.path.join(DIR, img[:-1]) + ".jpg"
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((300, 300), Image.ANTIALIAS)
            test_data.append([np.array(img), label])

    shuffle(test_data)
    return test_data

test_data = load_test_data()
testImages = np.array([i[0] for i in test_data]).reshape(-1, 300, 300, 1)
testImages = testImages/255
testLabels = np.array([i[1] for i in test_data])

train_data = load_training_data()
trainImages = np.array([i[0] for i in train_data]).reshape(-1, 300, 300, 1)
trainImages = trainImages/255
trainLabels = np.array([i[1] for i in train_data])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(96, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(trainImages, trainLabels, epochs=5
                    ,validation_data=(testImages, testLabels))
