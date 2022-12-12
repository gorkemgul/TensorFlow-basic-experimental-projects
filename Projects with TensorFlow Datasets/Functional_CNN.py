# Import dependencies 
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.activations import relu
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the mnist data and reshape it
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Create a Functional CNN Model
def CNN():
    inputs = tf.keras.Input(shape = (32,32,3))
    x = Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = relu(x)
    x = Conv2D(64, 5, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(10)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

model = CNN()

# Compile the model
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              optimizer = tf.keras.optimizers.Adam(lr=3e-4),
              metrics = ['accuracy'],)

BATCH_SIZE = 64
epochs = 10

# Fit the model 
model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = epochs, verbose = 2)
model.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

# Model summary
print(model.summary())