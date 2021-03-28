" ENPH 455 Thesis: Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies using Machine Learning Techniques "
" Regression Convolutional Neural Net to Determine the Central Halo Density "
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

# load the stored data
data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10K_Regress_Rho.npy', allow_pickle=True)
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
        self.dense1 = Dense(32, activation='relu')  # compute the first dense neural network layer
        # self.dense2 = Dense(32, activation='relu')  # compute the second dense neural network layer
        self.dense3 = Dense(1)
        self.dropout = Dropout(0.5) # conduct dropout of node to avoid overfitting

    
    # define the call structure for the model
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        # x = self.dense2(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense3(x)

    def model(self):
        x = Input(shape=galaxy_shape)
        return Model(inputs=[x], outputs=self.call(x))

# initialize the convolutional neural network model
Conv_Model = ConvModel()

#define the loss object and metrics
loss_object = tf.keras.losses.MeanSquaredError()  # define the MSE loss object
optimizer = tf.keras.optimizers.Adam()  # define the parametr optimizer Adam algorithm

# define the training metrics
train_MSE = tf.keras.metrics.MeanSquaredError(name='train_MSE')
train_MAE = tf.keras.metrics.MeanAbsoluteError(name='train_MAE')
train_MAPE = tf.keras.metrics.MeanAbsolutePercentageError(name='train_MAPE')

# define the testing metrics
test_MSE = tf.keras.metrics.MeanSquaredError(name='test_MSE')
test_MAE = tf.keras.metrics.MeanAbsoluteError(name='test_MAE')
test_MAPE = tf.keras.metrics.MeanAbsolutePercentageError(name='test_MAPE')

# define the tf function which will define the training steps for the model
@tf.function
def train_step(galaxies, labels):
    with tf.GradientTape() as tape:
        # training=True only needed if there are layers with different behaviour during training versus inference (ie. dropout)
        predictions = Conv_Model(galaxies, training=True)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, Conv_Model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Conv_Model.trainable_variables))

    train_MSE(labels, predictions)
    train_MAE(labels, predictions)
    train_MAPE(labels, predictions)
    

# define the tf function to test the model
@tf.function
def test_step(galaxies, labels):
    # training=False only needed if there are layers with different behaviour during training versus inference (ie. dropout)
    predictions = Conv_Model(galaxies, training=False)
    t_loss = loss_object(labels, predictions)

    test_MSE(labels, predictions)
    test_MAE(labels, predictions)
    test_MAPE(labels, predictions)

# set number of epochs to run the model over
EPOCHS = 100

print('Starting Epochs\n')
start = timeit.default_timer()

# establish results arrays
trainMSE = np.empty(EPOCHS)
trainMAE = np.empty(EPOCHS)
trainMAPE = np.empty(EPOCHS)

testMSE = np.empty(EPOCHS)
testMAE = np.empty(EPOCHS)
testMAPE = np.empty(EPOCHS)

epoch_num = np.arange(1, EPOCHS+1, step=1)

# start the training and testing loop
for epoch in range(EPOCHS):

    # reset the loss and accuracy metrics at start of each epoch
    train_MSE.reset_states()
    train_MAE.reset_states()
    train_MAPE.reset_states()

    test_MSE.reset_states()
    test_MAE.reset_states()
    test_MAPE.reset_states()

    # train the model
    for galaxies, labels in train_tensor:
        train_step(galaxies, labels)

    # test the model
    for t_galaxies, t_labels in test_tensor:
        test_step(t_galaxies, t_labels)

    # store training and testing metrics
    trainMSE[epoch] = train_MAE.result()
    trainMAE[epoch] = train_MAE.result()
    trainMAPE[epoch] = train_MAPE.result()

    testMSE[epoch] = test_MSE.result()
    testMAE[epoch] = test_MAE.result()
    testMAPE[epoch] = test_MAPE.result()

    # printing template for updates
    template = 'Epoch {}, MSE: {}, MAE: {}, MAPE: {}, Test MSE: {}, Test MAE: {}, Test MAPE: {}'
    print(template.format(epoch+1, train_MSE.result(), train_MAE.result(), train_MAPE.result(), test_MSE.result(), test_MAE.result(), test_MAPE.result()))

# general model architecture summary
Conv_Model.model().summary()

stop = timeit.default_timer()
print('Epochs Completed.  Time to run: ', stop-start)


# Save regression results to a csv file
# write metrics to dataframe
regress_results = pd.DataFrame({'Epoch': epoch_num, 'train_MSE': trainMSE, 'train_MAE': trainMAE, 
                                'train_MAPE': trainMAPE, 'test_MSE': testMSE, 'test_MAE': testMAE,
                                'test_MAPE': testMAPE})

# create save location folder if does not already exist
save_path = "./dSph_data/Rho_Regression_Results"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# write file name for regression results
currentDT = datetime.datetime.now()
currentDT = currentDT.strftime("%Y_%m_%d_%H_%M_%S_%f")
save_location = os.getcwd() + '\\dSph_data\\Rho_Regression_Results\\Rho_Regress_{}_'.format(EPOCHS) + currentDT + '.csv'

# save regression 
regress_results.to_csv(save_location, mode='a+')
