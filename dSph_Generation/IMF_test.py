import numpy as np
import matplotlib.pyplot as plt
from dSph_Model import chabrier_imf
import scipy.stats.distributions as st
import scipy as sci

if __name__ == "__main__":
    chbimf = chabrier_imf()
    sizeRVS = 100
    for N in range(sizeRVS):


    plt.figure()
    plt.plot(m,chbimf._pdf(m))
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    plt.pause(10)

    # # histogram on linear scale
    # hist, bins, _ = plt.hist(x, bins=8)
    # # histogram on log scale. 
    # # Use non-equal bin sizes, such that they look equal on log scale.
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    # plt.subplot(111)
    # plt.hist(x, bins=logbins)
    # plt.xscale('log')
    # plt.show()
    # plt.pause(10)