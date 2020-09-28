" ENPH 455 Thesis: Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies using Machine Learning Techniques "
" Classification Convolutional Neural Net to Classify the Dark Matter Distribution Profile as Cusp or Core "
# Convolutional Neural Net
# Robbie Faraday - 20023538
# Created March 19, 2020

# import necessary libraries
import os
import numpy as np 
import pandas as pd 
import timeit
import datetime
# disable the tensorflow logging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, Dropout
from tensorflow.keras import Model, Input

# check which device Tensors and operations are allocated to
# tf.debugging.set_log_device_placement(True)

data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10K_Classification.npy', allow_pickle=True)
print("Data Loaded")
galaxy_shape = (25, 25, 25, 1)

train_data = data[:7999]
train_features, train_labels = list(list(zip(*train_data))[0]), list(list(zip(*train_data))[1])
test_data = data[7999:]
test_features, test_labels = list(list(zip(*test_data))[0]), list(list(zip(*test_data))[1])

start = timeit.default_timer()

train_tensor = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(8000).batch(32)
test_tensor = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(32)

stop = timeit.default_timer()
print("Training and Test Tensors Generated, Time: ",stop-start)

# create the model class for the convolutional neural net classifier
class ConvModel(Model):
    # initialize the model structure
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = Conv3D(32, (3,3,3), activation='relu', input_shape=galaxy_shape, padding='same')  # compute first convolution
        self.conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')  # compute second convolution
        self.flatten = Flatten()  # flatten the data
        self.dense1 = Dense(128, activation='relu')  # compute the first dense neural network layer
        self.dense2 = Dense(2)  # compute the second dense neural network layer
        self.dropout = Dropout(0.5)
    
    # define the call structure for the model
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)

    def model(self):
        x = Input(shape=galaxy_shape)
        return Model(inputs=[x], outputs=self.call(x))

Conv_Model = ConvModel()

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# define the tf function which will define the training steps for the model
@tf.function
def train_step(galaxies, labels):
    with tf.GradientTape() as tape:
        # training=True only needed if there are layers with different behaviour during training versus inference (ie. dropout)
        predictions = Conv_Model(galaxies, training=True)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, Conv_Model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Conv_Model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# define the tf function to test the model
@tf.function
def test_step(galaxies, labels):
    # training=False only needed if there are layers with different behaviour during training versus inference (ie. dropout)
    predictions = Conv_Model(galaxies, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# set number of epochs to run the model over
EPOCHS = 25

print('Starting Epochs\n')
start = timeit.default_timer()

# establish results array
train_results = np.empty([EPOCHS, 2])
test_results = np.empty([EPOCHS, 2])

# start the training and testing loop
for epoch in range(EPOCHS):

    # reset the loss and accuracy metrics at start of each epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for galaxies, labels in train_tensor:
        train_step(galaxies, labels)

    for t_galaxies, t_labels in test_tensor:
        test_step(t_galaxies, t_labels)

    train_results[epoch,0] = train_loss.result()
    train_results[epoch,1] = train_accuracy.result()

    test_results[epoch,0] = test_loss.result()
    test_results[epoch,1] = test_accuracy.result()

    # printing template for updates
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1, train_loss.result()*100, train_accuracy.result()*100, test_loss.result()*100, test_accuracy.result()*100))

# generate summary of model architecture
Conv_Model.model().summary()

stop = timeit.default_timer()
print('Epochs Completed.  Time to run: ', stop-start)

class_results = pd.DataFrame({'train_loss': train_results[:,0], 'train_accuracy': train_results[:,1], 'test_loss': test_results[:,0], 'test_accuracy': test_results[:,1]})

# create save location folder if does not already exist
save_path = "./dSph_data/Classification_Results"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# write file name for regression results
currentDT = datetime.datetime.now()
currentDT = currentDT.strftime("%Y_%m_%d_%H_%M_%S_%f")
save_location = os.getcwd() + '\\dSph_data\\Classification_Results\\Classification_{}_'.format(EPOCHS) + currentDT + '.csv'

# save regression 
class_results.to_csv(save_location, mode='a+')