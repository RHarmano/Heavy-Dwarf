"""
Data Preprocessing for K-means Optimization

Author  : Roman Harman
Date    : September 30, 2020

Adapted for panadas data storage from dataPreprocessing_Classifier.py by Robbie Faraday
fit_transform call not necessary on data if using sklearn as fit_transform may be called in a pipeline
"""

# import necessary libraries
import os
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import timeit
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Read in the galaxy metadata
galaxy_data_dir = os.getcwd() + '/dSph_data/1000_stars_Mass_Burkert/'
metadata_labels = ['GalaxyId', 'nu_dist', 'N', 'L', 'r_half', 'DM_dist', 'r_core', 'rho_central', 'beta', 'D', 'DM_mass']
parameters_df = pd.DataFrame(columns = metadata_labels)

# data preprocessing function
def strip_data(data_dir, labels_df=parameters_df):
    galaxies = os.listdir(data_dir)
    galaxy_list = []
    count = 1
    for galaxy in galaxies:

        labels_temp = pd.read_csv(data_dir+str(galaxy), index_col = 0, nrows = 10, usecols = [0,1],  header = None).transpose()
        labels_temp.insert(0, 'GalaxyId', galaxy)
        labels_df = labels_df.append(labels_temp, ignore_index=True, sort=False)

        stars_df = pd.read_csv(data_dir+str(galaxy), index_col = 0, skiprows = 10, usecols = [0, 1, 2, 3])

        # Standardize the position and velocity values
        scaler = StandardScaler()
        stars_df[['theta_x', 'theta_y', 'z_velocity']] = scaler.fit_transform(stars_df[['theta_x', 'theta_y', 'z_velocity']]) # standardization
        stars_data = stars_df.to_numpy()

        # Generate 3D histogram of the data to bin the stars based on theta_x, theta_y, vz_velocity
        stars_hist, bin_div = np.histogramdd(stars_data, bins=50, density=False)

        stars_hist = np.expand_dims(stars_hist, axis=3)
        galaxy_list.append(stars_hist)

        if count % 100 == 0:
            print('Galaxies Processed: ', count)
        count += 1


    labels_df[['rho_central', 'r_core', 'DM_mass']] = scaler.fit_transform(labels_df[['rho_central', 'r_core', 'DM_mass']])

    rho_list = labels_df['rho_central'].to_list()
    core_list = labels_df['r_core'].to_list()
    mass_list = labels_df['DM_mass'].to_list()
    
    labels_save = os.getcwd() + '\\dSph_data\\labels_data_standard.csv'

    labels_df.to_csv(labels_save, mode='a+')
    
    return galaxy_list, rho_list, core_list, mass_list,


# create processed data file location if it does not exist
if not os.path.isdir("./dSph_data/ProcessedData"):
        os.mkdir("./dSph_data/ProcessedData")

#call preprocessing function to process the raw data
galaxy_list, rho_list, core_list, mass_list = strip_data(galaxy_data_dir)

# compile data and labels into list of tuples
galaxy_data_rho = list(zip(galaxy_list, rho_list))  # data to predict central halo density
galaxy_data_core = list(zip(galaxy_list, core_list))  # data to predict core radius 
galaxy_data_mass = list(zip(galaxy_list, mass_list))  # data to predict DM mass

# save the dataframe for later access
np.save('./dSph_data/ProcessedData/dSph_Processed_Data_KMeans_rho', galaxy_data_rho)
np.save('./dSph_data/ProcessedData/dSph_Processed_Data_KMeans_core', galaxy_data_core)
np.save('./dSph_data/ProcessedData/dSph_Processed_Data_KMeans_mass', galaxy_data_mass)
