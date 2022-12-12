# Import dependencies 
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the mnist data and reshape it
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Creating a Sequential CNN Model
model = Sequential([
    Conv2D(32, 3, padding = 'valid', activation = 'relu'), #valid is the defult
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(64, 3, activation = 'relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation = 'relu'),
    Flatten(),
    Dense(64, activation = 'relu'),
    Dense(10)
    ])

# Compile the model 
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = Adam(lr=3e-4),
              metrics = ['accuracy'],)

# Set up the parameters and fit the model
BATCH_SIZE = 64
epochs = 10

model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = epochs, verbose = 2)
model.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

# Print out the model summary
print(model.summary())