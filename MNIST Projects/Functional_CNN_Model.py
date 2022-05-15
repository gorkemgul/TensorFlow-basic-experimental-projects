# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
#%%
#Loading mnist data and reshaping
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
#%%
#Creating Functional CNN Model
def CNNMOdel():
    inputs = tf.keras.Input(shape = (32,32,3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv2D(64, 5, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(10)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

model = CNNMOdel()
#%%
#Model Compiling
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = tf.keras.optimizers.Adam(lr=3e-4),
              metrics = ['accuracy'],)
#%%
batch_size = 64
epoch = 10
#model fit
model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch, verbose = 2)
model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 2)
#%%
#model summary
print(model.summary())