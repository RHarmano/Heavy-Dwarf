" ENPH 455 Thesis: Reproducing the Dark Matter Structure of Simulated Dwarf Spheroidal Galaxies using Machine Learning Techniques "
# Data Preprocessing
# Robbie Faraday - 20023538
# Created February 7, 2020

# Import desired libraries
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import timeit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Establish plot design parameters
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('axes', labelsize=14)     # fontsize of the axes labels
plt.rc('axes', titlesize=14)     # fontsize for title

# Read in the galaxy data
data_location = os.getcwd() + '\\dSph_test_1000.csv'
struct_dataframe = pd.read_csv(data_location, nrows=8)
stars_dataframe = pd.read_csv(data_location, skiprows=9, usecols=[1, 2, 3])

# Standardize the position and velocity values
scaler = StandardScaler()
minmax = MinMaxScaler(feature_range=(-1,1))
stars_dataframe[['theta_x', 'theta_y', 'z_velocity']] = scaler.fit_transform(stars_dataframe[['theta_x', 'theta_y', 'z_velocity']]) # standardization, zero-mean
stars_dataframe[['theta_x', 'theta_y', 'z_velocity']] = minmax.fit_transform(stars_dataframe[['theta_x', 'theta_y', 'z_velocity']]) # min-max restrict range to [-1,1]

stars_data = stars_dataframe.to_numpy()

# Generate 3D histogram of the data to bin the stars based on theta_x, theta_y, vz_velocity
stars_hist, bin_div = np.histogramdd(stars_data, bins=10, density=False)

vmax=stars_hist.max()

# Create 3D figure to show all the points of data
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111, projection="3d")
ax1.plot(stars_data[:,0], stars_data[:,1], stars_data[:,2], 'k.', alpha = 0.7)

# Generate the meshgrid to show the bin edges
X, Y = np.meshgrid(bin_div[0][:-1], bin_div[1][:-1])

# Create slices to show histogram concentration
for ct in [2, 5, 8]:
    cs = ax1.contourf(X, Y, stars_hist[:,:,ct], zdir='z', offset=bin_div[2][ct], cmap=plt.cm.RdYlBu_r, alpha=0.5)

cbar = fig.colorbar(cs)
plt.locator_params(nbins=5)
ax1.set_xlim(-1.1,1.1)
ax1.set_xlabel("$\\theta_x$", labelpad=10)
ax1.set_ylim(-1.1,1.1)
ax1.set_ylabel("$\\theta_y$", labelpad=10)
ax1.set_zlim(-1.1,1.1)
ax1.set_zlabel("$V_z$", labelpad=10)
ax1.view_init(elev=20., azim=50.)
plt.tight_layout()
# uncomment line below to save the plot as a PDF
# plt.savefig('./binned_stars_1000.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
