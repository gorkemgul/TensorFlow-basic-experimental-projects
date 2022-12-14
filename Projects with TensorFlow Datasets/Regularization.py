# Import dependencies
import tensorflow as tf
from keras import Model, regularizers, activations
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.losses import SparseCategoricalCrossentropy
from keras.datasets import cifar10

# Set up the device
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the data and rescale it
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, y_train = X_train.astype('float32') / 255.0, y_train.astype('float32') / 255.0

# Create the CNN model and add L2 Regularization
def CNN_Regularization():
    inputs = Input(shape = (32, 32, 3))
    x = Conv2D(32, 3, padding = "same", kernel_regularizer = regularizers.L2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, padding = "same", kernel_regularizer = regularizers.L2(0.01))(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, padding = "same", kernel_regularizer = regularizers.L2(0.01))(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', kernel_regularizer = regularizers.L2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10)(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model

cnn_model_with_regularization = CNN_Regularization()

# Set up the hyper-parameters
BATCH_SIZE = 64
epochs = 150

# Compile and Fit the model
cnn_model_with_regularization.compile(loss = SparseCategoricalCrossentropy(from_logits = True),
                                      optimizer = tf.keras.optimizers.Adam(lr = 3e-4),
                                      metrics = ['accuracy'])

cnn_model_with_regularization.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = epochs, verbose = 2)

# Evaluate the model
cnn_model_with_regularization.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose = 2)