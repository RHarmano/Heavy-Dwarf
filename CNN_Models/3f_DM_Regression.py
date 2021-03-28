"""
ENPH 455 Thesis Project: 
Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies
using Machine Learning Techniques

Author  : Roman Harman
Date    : September 30, 2020

Summary
-------
Unsupervised learning methods: KMeans algorithm. 
"""

# import necessary libraries
import os
import numpy as np 
import pandas as pd 
import timeit
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

normalize_accuracy = True

# Numpy sometimes has trouble loading in arrays from a different directory, so ensure pickle is allowed
rho_data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10000_KMeans_rho.npy', allow_pickle=True)
mass_data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10000_KMeans_mass.npy', allow_pickle=True)
core_data = np.load('./dSph_data/ProcessedData/dSph_Processed_Data_10000_KMeans_core.npy', allow_pickle=True)

data = pd.DataFrame({'rho':rho_data[:,1], 'mass': mass_data[:,1], 'core': core_data[:,1]})

print("Data loaded has dimensions: {} samples, {} features".format(str(len(data.index)),str(len(data.columns))))
train_data, test_data = train_test_split(data, train_size=0.8)

# clf = MLPRegressor(hidden_layer_sizes=128, activation='relu', solver='adam', 
#                     alpha = 10**(-5), batch_size='auto', learning_rate='constant', max_iter=100)
clf = svm.SVC()
clf.fit(train_data[['mass','core','rho']], np.ones(shape=(len(train_data.index),1)))
prediction = clf.predict(test_data[['mass','core','rho']])
test_accuracy = clf.score(test_data[['mass','core']], test_data['rho'])
# accuracy = accuracy_score(test_data['rho'], prediction, normalize=normalize_accuracy)
print("{}Testing Accuracy: {}".format("Normalized " if normalize_accuracy == True else "",
                                                                test_accuracy))