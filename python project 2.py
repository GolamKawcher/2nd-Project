# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:47:12 2023

@author: HP
"""

import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from google.colab import files
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#to import dataset from google apis
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (evaluating_images, evaluating_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images_reshaped = train_images.reshape(60000, 28, 28, 1)

def task_information():
    print("This program trains multiple convolutional neural network models with varying hyperparameters and saves the results in a CSV file.")

def build_model(num_layers, model_number, input_shape=(28, 28, 1)):
    filters = []
    kernel_sizes = []
    activations = []
    pooling_sizes = []
    paddings = []
    i = 1

    epochs = int(input("Enter the number of epochs for Model {}: ".format(model_number)))
    print("")

    while i < num_layers+1:
        filter_value = int(input("Enter filter size for {} layer: ".format(i)))
        filters.append(filter_value)

        kernel_size_value = int(input("Enter kernel size for {} layer: ".format(i)))
        kernel_sizes.append(kernel_size_value)

        activation_value = input("Enter activation function for {} layer: ".format(i))
        activations.append(activation_value)

        pooling_size_value = int(input("Enter pooling size for {} layer: ".format(i)))
        pooling_sizes.append(pool_size_value)

        padding_value = input("Enter the padding value for {} layer: ".format(i))
        paddings.append(padding_value)

        print()
        i += 1

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

    for filter, kernel_size, activation, pool_size, padding, i in zip(filters, kernel_sizes, activations, pooling_sizes, paddings, range(1, num_layers+1)):
        model.add(Conv2D(filters=filter, kernel_size=(kernel_size,kernel_size), activation=activation, padding=padding))
        model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images_reshaped, train_labels, epochs=epochs, validation_split=0.4)
    return history, model, epochs

def main():
    task_information()
    print("")

    num_models = int(input("Enter the number of models: "))
    print("")

    df_columns = ['Model Name', 'Architecture', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss', 'Epochs']
    dataframe = pd.DataFrame(columns=df_columns)

    for i in range(1, num_models+1):
        print("Enter the details of architecture and hyperparameters for Model {}".format(i))
        layers = int(input("Enter the layers for Model {}: ".format(i)))
        print("")
        history, model
