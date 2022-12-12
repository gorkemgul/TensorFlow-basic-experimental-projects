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
#Creating Sequential Model
model = Sequential([
    Dense(512, activation = tf.nn.relu),
    Dense(256, activation = tf.nn.relu),
    Dense(10)
])
#%% Model Compiling
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
#%% model fit
batch_size = 32
epoch = 5
model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch, verbose = 2)
model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 2)
#%% model summary
print(model.summary())