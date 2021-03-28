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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# create processed data file location if it does not exist
if not os.path.isdir("./Final_Figs"):
        os.mkdir("./Final_Figs")


# set up matplotlib text parameters
plt.rc('xtick', labelsize=14)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the y tick labels
plt.rc('axes', labelsize=18)     # fontsize of the axes labels
plt.rc('axes', titlesize=18)     # fontsize for title
plt.rc('xtick.major', pad=8)     # pad between x tick labels and axis labels
plt.rc('ytick.major', pad=8)     # pad between y tick labels and axis labels

# Read in the galaxy metadata
data_dir1 = os.getcwd() + '/dSph_data/labels_data.csv'
labels_df1 = pd.read_csv(data_dir1, header=0, index_col=0)

data_dir2 = os.getcwd() + '/dSph_data/labels_data_class.csv'
labels_df2 = pd.read_csv(data_dir2, header=0, index_col=0)

labels_df2 = labels_df2[labels_df2['DM_dist'] == 'NFW']
print(labels_df2.head(100))

def plot_hist(dataframe, data, filename, xlabel):

    in_data = dataframe[data].to_numpy()
    in_data = in_data.astype(float)
    fig, ax = plt.subplots(figsize=(10,6))

    ax.grid(axis='y', alpha=0.7)
    ax.set(xlabel=xlabel, ylabel='Frequency', title=filename)

    n, bins, pathces = ax.hist(x=in_data, bins=20, alpha=0.85, rwidth=0.85)
    
    fig.tight_layout()
    plt.savefig('./Final_Figs/' + filename + '.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()

plot_hist(labels_df2, 'M_200', 'Characteristic Mass Histogram', 'Characteristic Mass [$M_{solar}$]')


