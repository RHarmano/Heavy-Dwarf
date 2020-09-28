import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 



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

class_results = pd.read_csv(os.getcwd()+'./dSph_data/Classification_Results/Classification_25_2020_04_14_18_54_47_769255.csv', header=0, index_col=0)

mass_8 = pd.read_csv(os.getcwd()+'./dSph_data/Mass_Regression_Results_1_Layer/Mass_Regress_100_2020_04_10_15_48_29_042235.csv', header=0, index_col=0)
mass_16 = pd.read_csv(os.getcwd()+'./dSph_data/Mass_Regression_Results_1_Layer/Mass_Regress_100_2020_04_10_13_57_47_034554.csv', header=0, index_col=0)
mass_32 = pd.read_csv(os.getcwd()+'./dSph_data/Mass_Regression_Results_1_Layer/Mass_Regress_100_2020_04_10_17_44_36_464758.csv', header=0, index_col=0)
mass_64 = pd.read_csv(os.getcwd()+'./dSph_data/Mass_Regression_Results_1_Layer/Mass_Regress_100_2020_04_10_20_25_15_227642.csv', header=0, index_col=0)
mass_128 = pd.read_csv(os.getcwd()+'./dSph_data/Mass_Regression_Results_1_Layer/Mass_Regress_100_2020_04_11_03_04_02_106181.csv', header=0, index_col=0)

rho_16 = pd.read_csv(os.getcwd()+'./dSph_data/Rho_Regression_Results/Rho_Regress_100_2020_04_06_20_41_14_170067.csv', header=0, index_col=0)

def plot_regress(regress_results):

    train_MSE = regress_results['train_MSE'].to_numpy()
    train_MAE = regress_results['train_MAE'].to_numpy()

    test_MSE = regress_results['test_MSE'].to_numpy()
    test_MAE = regress_results['test_MAE'].to_numpy()

    epochs=np.arange(1,len(train_MSE)+1, 1)

    fig, (ax1,ax2) = plt.subplots(2, figsize=(8,6), sharex=True)
    ax1.set_title('Regression Performance Progression (16 Nodes)')

    ax1.set(ylabel='Mean Squared Error')
    ax2.set(ylabel='Mean Absolute Error', xlabel='Epochs')

    ax1.plot(epochs, train_MSE, color='b', lw=2)
    ax1.plot(epochs, test_MSE, color='r', lw=2)
    ax1.legend(['Train MSE', 'Test MSE'], loc='upper right', fancybox = True, fontsize = 'x-large', framealpha = 0.7)

    ax2.plot(epochs, train_MAE, color='b', lw=2)
    ax2.plot(epochs, test_MAE, color='r', lw=2)
    ax2.legend(['Train MAE', 'Test MAE'], loc='upper right', fancybox = True, fontsize = 'x-large', framealpha = 0.7)

    fig.tight_layout()
    plt.savefig('./Final_Figs/Rho_16_Results.pdf', format='pdf', dpi=1200, bbox_inches='tight')

    plt.show()

plot_regress(rho_16)


