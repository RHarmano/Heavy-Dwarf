" ENPH 455 Thesis: Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies using Machine Learning Techniques "
# Data Preprocessing
# Robbie Faraday - 20023538
# Created March 10, 2020

# import necessary libraries
import os
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import timeit
from sklearn.preprocessing import StandardScaler


# Read in the galaxy metadata
data_dir = os.getcwd() + '/dSph_data/1000_stars_Mix/'
galaxies = os.listdir(data_dir)
label_names = ['GalaxyId', 'nu_dist', 'N', 'L', 'r_half', 'DM_dist', 'r_core', 'rho_central', 'beta', 'D']
labels_df = pd.DataFrame(columns = label_names)

# data preprocessing function
def ProcessData(data_dir, galaxy, labels_df=labels_df):
    labels_temp = pd.read_csv(data_dir+str(galaxy), index_col = 0, nrows=9, usecols = [0,1],  header = None).transpose()
    labels_temp.insert(0, 'GalaxyId', galaxy)
    labels_df = labels_df.append(labels_temp, ignore_index=True, sort=False)

    
    if labels_temp.at[1, 'DM_dist'] == 'burkert':
        label = np.array([0,1])
    
    elif labels_temp.at[1, 'DM_dist'] == 'NFW':
        label = np.array([1,0])


    stars_df = pd.read_csv(data_dir+str(galaxy), index_col = 0, skiprows=9, usecols=[0, 1, 2, 3])
    # Standardize the position and velocity values
    scaler = StandardScaler()
    stars_df[['theta_x', 'theta_y', 'z_velocity']] = scaler.fit_transform(stars_df[['theta_x', 'theta_y', 'z_velocity']]) # standardization, zero-mean

    stars_data = stars_df.to_numpy()

    # Generate 3D histogram of the data to bin the stars based on theta_x, theta_y, vz_velocity
    stars_hist, bin_div = np.histogramdd(stars_data, bins=25, density=False)

    labels_save = os.getcwd() + '\\dSph_data\\labels_data_class.csv'

    labels_df.to_csv(labels_save, mode='a+')

    return stars_hist, label
# create processed data file location if it does not exist
if not os.path.isdir("./dSph_data/ProcessedData"):
        os.mkdir("./dSph_data/ProcessedData")

# compiled galaxy data
galaxy_data = []
count = 1
for galaxy in galaxies:

    #call preprocessing function to process the raw data
    stars_hist, label = ProcessData(data_dir, galaxy)
    stars_hist = np.expand_dims(stars_hist, axis=3)    
    galaxy_data.append([stars_hist, label])

    if count % 100 == 0:
        print('Galaxies Processed: ', count)
    
    count +=1

# save the processed data to a single file
np.save('./dSph_data/ProcessedData/dSph_Processed_Data_10K_Classification', galaxy_data)
 