" ENPH 455 Thesis: Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies using Machine Learning Techniques "
" Regression Convolutional Neural Net to Determine the Galaxy Dark Matter Mass "
# Convolutional Neural Net
# Robbie Faraday - 20023538
# Created April 9, 2020

# import necessary libraries
import os
import numpy as np 
import pandas as pd 
import timeit
import datetime

# disable the tensorflow logging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, Dropout, BatchNormalization
from tensorflow.keras import Model, Input, regularizers

# load the stored data
data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10K_Regress_Mass.npy', allow_pickle=True)
print("Data Loaded")
galaxy_shape = (25, 25, 25, 1)
STEPS_PER_EPOCH = 8000/32  # size of training data set divided by the batch size

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

        self.conv1 = Conv3D(32, (3,3,3), activation='relu', input_shape=galaxy_shape, kernel_regularizer=regularizers.l2(0.001), padding='same')  # compute first convolution
        self.batch1 = BatchNormalization()
        self.maxpool1 = MaxPool3D((2,2,2))  # conduct max pooling of data
        self.conv2 = Conv3D(32, (3,3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same')  # compute second convolution
        self.batch2 = BatchNormalization()
        self.maxpool2 = MaxPool3D((2,2,2))  # conduct max pooling of data
        self.flatten = Flatten()  # flatten the data
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))  # compute the first dense neural network layer
        self.batch3 = BatchNormalization()
        self.dropout1 = Dropout(0.5)  # conduct dropout of node to avoid overfitting
        # self.dense2 = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001))  # compute the second dense neural network layer
        # self.dropout2 = Dropout(0.5)  # conduct dropout of node to avoid overfitting
        self.dense_output = Dense(1)  # output layer
        

    
    # define the call structure for the model
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch3(x)
        if training:
            x = self.dropout1(x, training=training)
        # x = self.dense2(x)
        # if training:
        #     x = self.dropout2(x, training=training)
        return self.dense_output(x)


# initialize the convolutional neural network model
Conv_Model = ConvModel()

# define optimizer function
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH*10, decay_rate=1, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr_schedule)  # define the parametr optimizer Adam algorithm

# compile the model
Conv_Model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae', 'mape'])

# set number of epochs to run the model over
EPOCHS = 1

print('Starting Epochs\n')
start = timeit.default_timer()

# train the model
history = Conv_Model.fit(train_tensor, epochs = EPOCHS, validation_data = test_tensor)

# conduct and evaluation of the test data using the trained model
loss, MSE, MAE, MAPE = Conv_Model.evaluate(test_tensor)
print("Test Set MSE: {}, MAE: {}, MAPE: {}".format(MSE, MAE, MAPE))

# predict the dark matter density to test the accuracy of the model
test_predictions = Conv_Model.predict(test_tensor).flatten()

prediction_data = pd.DataFrame({'test_labels': test_labels, 'prediction_labels': test_predictions})

# general model architecture summary
Conv_Model.model().summary()


stop = timeit.default_timer()
print('Epochs Completed.  Time to run: ', stop-start)

# Save regression results to a csv file
# write metrics to dataframe
regress_results = pd.DataFrame(history.history)
regress_results['epoch'] = history.epoch


# create save location folder if does not already exist
save_path = "./dSph_data/Mass_Regression_Results_Overfit"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# # write file name for regression results
# currentDT = datetime.datetime.now()
# currentDT = currentDT.strftime("%Y_%m_%d_%H_%M_%S_%f")
# save_location = os.getcwd() + '\\dSph_data\\Mass_Regression_Results_Overfit\\Mass_Regress_{}_'.format(EPOCHS) + currentDT + '.csv'

# # write prediction data to a file
# save_predictions = os.getcwd() + '\\dSph_data\\Mass_Regression_Results_Overfit\\Mass_Predictions_{}_'.format(EPOCHS) + currentDT + '.csv'


# # save regression and predictions
# regress_results.to_csv(save_location, mode='a+')
# prediction_data.to_csv(save_predictions, mode='a+')