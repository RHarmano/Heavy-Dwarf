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
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, Dropout
from tensorflow.keras import Model, Input


# load the stored data
data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10K_Regress_Mass.npy', allow_pickle=True)
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
tf.conv
# build sequential model using the Keras API
def build_model():
    # define the model layers using the sequential API
    model = tf.keras.models.Sequential()
    # model.add(Conv3D(32, (3,3,3), activation = 'relu', input_shape = galaxy_shape, padding = 'same'))  # compute first convolution
    # model.add(Conv3D(64, (3,3,3), activation = 'relu', padding = 'same'))  # compute second convolution
    model.add(Flatten())  # flatten the convolution data
    model.add(Dense(16, activation = 'relu', input_shape=(15625,)))
    # model.add(Dropout(0.5))  # conduct dropout of nodes to avoid overfitting
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae', 'mape'])

    return model

Conv_Model = build_model()

EPOCHS = 25
history = Conv_Model.fit(train_tensor, epochs = EPOCHS, validation_data = test_tensor)

Conv_Model.summary()

loss, MSE, MAE, MAPE = Conv_Model.evaluate(test_tensor)
print("Test Set MSE: {}, MAE: {}, MAPE: {}".format(MSE, MAE, MAPE))

stop = timeit.default_timer()
print('Epochs Completed.  Time to run: ', stop-start)


# # Save regression results to a csv file
# # write metrics to dataframe
# regress_results = pd.DataFrame({'Epoch': epoch_num, 'train_MSE': trainMSE, 'train_MAE': trainMAE, 
#                                 'train_MAPE': trainMAPE, 'test_MSE': testMSE, 'test_MAE': testMAE,
#                                 'test_MAPE': testMAPE})

# # create save location folder if does not already exist
# save_path = "./dSph_data/Mass_Regression_Results"
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)

# # write file name for regression results
# currentDT = datetime.datetime.now()
# currentDT = currentDT.strftime("%Y_%m_%d_%H_%M_%S_%f")
# save_location = os.getcwd() + '\\dSph_data\\Mass_Regression_Results\\Mass_Regress_{}_'.format(EPOCHS) + currentDT + '.csv'

# # save regression 
# regress_results.to_csv(save_location, mode='a+')
