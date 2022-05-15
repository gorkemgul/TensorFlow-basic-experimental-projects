#%% Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
#%% Loading mnist data and reshaping
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
#%%
#Creating Functional Model
inputs = tf.keras.Input(shape = (784))
x = Dense(512, activation = 'relu')(inputs)
x = Dense(256, activation = 'relu')(x)
outputs = Dense(10, activation = 'softmax')(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs)
#%% Model Compiling
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              optimizer = tf.keras.optimizers.Adam(lr=0.001),
              metrics = ['accuracy'])
#%% model fit
batch_size = 32
epoch = 5
model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch, verbose = 2)
model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 2)
#%% model summary
print(model.summary())