# Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras import Input, Model
from tensorflow.keras.datasets import mnist

# In case using a GPU 
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the mnist data and reshape it
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Create a Functional Model
def func_model():
    inputs = Input(shape = (784))
    x = Dense(512, activation = 'relu')(inputs)
    x = Dense(256, activation = 'relu')(x)
    outputs = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model

model = func_model()

# Compile the model
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              optimizer = tf.keras.optimizers.Adam(lr=0.001),
              metrics = ['accuracy'])

# Define the params and fit the model 
BATCH_SIZE = 32
epochs = 5

model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = epochs, verbose = 2)
model.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

# Print out the model summary
print(f"Model Summary:\n {model.summary()}")